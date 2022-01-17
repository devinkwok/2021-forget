import time
from collections import defaultdict
import torch
from torch import nn, optim
from torch.utils.data import Subset, DataLoader
from forget.training import measureforget


class train:
    def __init__(self, job, replicate_num):
        # structure of directory is eg ../jobs/job1/model1/
        self.job = job
        self.replicate_num = replicate_num
        self.model_dir = f"model{self.replicate_num}"
        self.loss = nn.CrossEntropyLoss()
        self.data = self.job.get_train_dataset()
        # metrics: load eval dataset to CUDA
        examples, _ = zip(*self.job.get_eval_dataset())
        self.eval_x = torch.stack(examples, dim=0).cuda()

    def example_orders(self):
        def randperm():
            return torch.randperm(self.job.n_train_examples)

        for i in range(self.job.n_epochs):
            if self.job.hparams["example order"] == "random":
                yield self.job.cached(
                    randperm, "rand_example_order", f"example_idx_epoch={i+1}.pt"
                )
            else:
                yield torch.arange(self.job.n_train_examples)

    def get_dataloader(self, train_data, order):
        shuffled_data = Subset(train_data, order)
        return DataLoader(shuffled_data, batch_size=self.job.batch_size, num_workers=1)

    def trainLoop(self):
        def train_epoch():  # name function so that job.cached() prints name
            return self.train_epoch(None, None, 0)

        ckpt = self.job.cached(train_epoch, self.model_dir, f"epoch={0}.pt")
        # epoch 0 is to save checkpoint at initialization
        for i, order in enumerate(self.example_orders()):
            epoch = i + 1
            dataloader = self.get_dataloader(self.data, order)
            # redefine function with previous ckpt as input
            def train_epoch():
                return self.train_epoch(ckpt, dataloader, epoch)

            # update with next ckpt
            ckpt = self.job.cached(train_epoch, self.model_dir, f"epoch={epoch}.pt")

    def train_epoch(self, ckpt, dataloader, epoch):
        model, optimizer, scheduler = self.get_model_optim(ckpt, epoch)
        batch_loss, batch_acc, eval_logits = [], [], []
        if (
            dataloader is None
        ):  # if no data, initialize model, run eval, and return ckpt
            model.eval()
            eval_logits.append(model(self.eval_x).detach().cpu())
        else:  # otherwise, train ckpt for 1 epoch on dataloader
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
                model.eval()
                eval_logits.append(model(self.eval_x).detach().cpu())
                # stats
                batch_loss.append(J.item())
                batch_acc.append(
                    y.eq(logits.detach().argmax(dim=1).cpu()).float().mean()
                )
                print(
                    f"\t{epoch}:{iter}\ta={batch_acc[-1]:0.2f} l={batch_loss[-1]:0.4f} t={time.perf_counter() - t_1:0.4f}"
                )
        # save per-iteration probabilities
        self.job.save_obj_to_subdir(
            torch.stack(eval_logits, axis=0), self.model_dir, f"eval_logits={epoch}.pt"
        )
        # return ckpt
        return self.checkpoint(model, optimizer, scheduler, batch_loss, batch_acc)

    def get_model_optim(self, ckpt: dict, epoch=0):
        if ckpt is None:
            model = self.job.get_model()
        else:
            assert "model_state_dict" in ckpt
            model = self.job.get_model(state_dict=ckpt["model_state_dict"])

        if self.job.hparams["model parameters"] == "default":
            optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, [], last_epoch=-1
            )  # equivalent to fixed lr
        elif self.job.hparams["model parameters"] == "resnet20":
            # hparams for resnet20/cifar10 from Frankle et al. Linear Mode Connectivity and the Lottery Ticket Hypothesis
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            # schedule lr drop at 32k and 48k iters
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [82, 123], 0.1)
        else:
            raise ValueError(
                "Invalid model parameters", self.job.hparams["model parameters"]
            )

        if ckpt is not None:
            assert "optimizer_state_dict" in ckpt and "scheduler_state_dict" in ckpt
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
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

    def evaluate_model(self, model, dataloader):
        model.eval()
        for x, y in dataloader:
            x = x.cuda()
            with torch.no_grad():
                output = model(x)
            yield output.detach(), y.detach()
        model.train()
