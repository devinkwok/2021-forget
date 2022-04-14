import time
import torch
from torch import nn, optim
from torch.utils.data import Subset, DataLoader

from forget.job import Job


class train:
    def __init__(self, job: Job):
        # structure of directory is eg ../jobs/job1/model1/
        self.job = job
        self.loss = nn.CrossEntropyLoss()
        self.data = self.job.get_train_dataset()
        # metrics: load eval dataset to CUDA
        examples, _ = zip(*self.job.get_eval_dataset())
        self.eval_x = torch.stack(examples, dim=0).cuda()

    def example_orders(self, replicate_dir):
        def randperm():
            return torch.randperm(self.job.n_train_examples)

        for i in range(self.job.n_epochs):
            if self.job.hparams["example order"] == "random_iid":
                yield self.job.cached(
                    randperm, replicate_dir, f"example_order_ep{i+1}.pt"
                )
            elif self.job.hparams["example order"] == "shuffled_fixed":
                yield self.job.cached(
                    randperm, "fixed_example_order", f"example_order_ep{i+1}.pt"
                )
            else:
                yield torch.arange(self.job.n_train_examples)

    def get_dataloader(self, train_data, order):
        shuffled_data = Subset(train_data, order)
        return DataLoader(shuffled_data, batch_size=self.job.batch_size, num_workers=1)

    def is_finished(self, replicate_dir):
        return self.job.file_exists(replicate_dir, f"ckpt-ep{self.job.n_epochs}.pt")

    def trainLoop(self, replicate_dir, fixed_init=None):
        def train_epoch():  # name function so that job.cached() prints name
            model, optimizer, scheduler = self.get_model_optim(
                {
                    "model_state_dict": fixed_init,
                    "optimizer_state_dict": None,
                    "scheduler_state_dict": None,
                }
            )
            return self.checkpoint(model, optimizer, scheduler, [], [])

        # epoch 0 is to save checkpoint at initialization
        ckpt = self.job.cached(train_epoch, replicate_dir, f"ckpt-ep{0}.pt")
        # run eval after caching so that logits for epoch 0 can be regenerated without changing model init
        model, _, _ = self.get_model_optim(ckpt)
        self.job.save_obj_to_subdir(
            torch.stack([self.eval(model)], axis=0), replicate_dir, f"logits-ep{0}.pt"
        )

        for i, order in enumerate(self.example_orders(replicate_dir)):
            epoch = i + 1
            dataloader = self.get_dataloader(self.data, order)
            # redefine function with previous ckpt as input
            def train_epoch():
                return self.train_epoch(ckpt, dataloader, epoch, replicate_dir)

            # update with next ckpt
            ckpt = self.job.cached(train_epoch, replicate_dir, f"ckpt-ep{epoch}.pt")

    def train_epoch(self, ckpt, dataloader, epoch, replicate_dir):
        model, optimizer, scheduler = self.get_model_optim(ckpt)
        batch_loss, batch_acc, eval_logits = [], [], []
        # if no data, initialize model and return ckpt
        print(
            f'{replicate_dir} ep={epoch} lr={optimizer.state_dict()["param_groups"][0]["lr"]}'
        )
        # otherwise, train ckpt for 1 epoch on dataloader
        for iter, batch in enumerate(dataloader):
            t_1 = time.perf_counter()
            # forward pass
            x, y = batch
            model.train()
            logits = model(x.cuda())
            # backward pass
            J = self.loss(logits, y.cuda())
            model.zero_grad()
            J.backward()
            optimizer.step()
            # eval on test data
            eval_logits.append(self.eval(model))
            # stats
            batch_loss.append(J.item())
            batch_acc.append(
                y.eq(logits.detach().argmax(dim=1).cpu()).float().mean().item()
            )
            print(
                f"\t{epoch}:{iter}\ta={batch_acc[-1]:0.2f} l={batch_loss[-1]:0.4f} t={time.perf_counter() - t_1:0.4f}"
            )
            if iter == epoch:  # pick one iter to save a sample batch from
                self.job.save_obj_to_subdir(
                    batch, replicate_dir, f"samplebatch_ep{epoch}_it{iter}.pt"
                )
        # save per-iteration probabilities
        self.job.save_obj_to_subdir(
            torch.stack(eval_logits, axis=0), replicate_dir, f"logits-ep{epoch}.pt"
        )
        scheduler.step()
        # return ckpt
        return self.checkpoint(model, optimizer, scheduler, batch_loss, batch_acc)

    def eval(self, model):
        model.eval()
        with torch.no_grad():
            eval_logit = model(self.eval_x).detach().cpu()
        return eval_logit

    def get_model_optim(self, ckpt: dict):
        # if "model_state_dict" is None, this will create a new random init
        model = self.job.get_model(state_dict=ckpt["model_state_dict"])

        if self.job.hparams["dataset"] == "CIFAR10" or self.job.hparams["dataset"] == "CIFAR100":
            # hparams for cifar10 from Frankle et al. Training Batchnorm and only Batchnorm
            optimizer = optim.SGD(
                model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
            )
            # schedule lr drop at 32k and 48k iters
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [80, 120], 0.1)
        else:
            raise ValueError("Invalid model parameters", self.job.hparams["dataset"])

        if ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return model, optimizer, scheduler

    def checkpoint(self, model, optimizer, scheduler, loss, accuracy):
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
            "train accuracy": accuracy,
        }
