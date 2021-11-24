import time
import torch
from torch import nn, optim
from forget.training import measureforget


class train:
    def __init__(self, job, replicate_num):
        # structure of directory is eg ../jobs/job1/model1/
        self.job = job
        self.replicate_num = replicate_num
        self.model_dir = f"model{self.replicate_num}"
        self.batch_size = int(
            self.job.hparams["batch size"]
        )  # note that this also gets passed to measureForget
        self.loss = nn.CrossEntropyLoss()
        self.dataloader = self.job.get_dataloader(train=True)
        # metrics: load eval dataset to CUDA
        examples, _ = zip(*self.job.get_eval_dataset())
        self.eval_x = torch.stack(examples, dim=0).cuda()

        self.forget_msrmt = measureforget.measureForget(
            self.job.n_epochs,
            num_batches=len(self.dataloader),
            batch_size=self.batch_size,
        )

    def trainLoop(self):
        ckpt = None
        # epoch 0 is to save checkpoint at initialization
        for epoch in range(0, self.job.n_epochs + 1):

            def train_epoch():  # name function so that job.cached() prints name
                return self.train_epoch(ckpt, epoch)

            ckpt = self.job.cached(train_epoch, self.model_dir, f"epoch={epoch}.pt")

        self.job.cached(
            lambda: self.forget_msrmt.forgetStatistics,
            self.model_dir,
            f"forgetstatsepoch={epoch}.pt",
        )
        self.job.cached(
            lambda: self.forget_msrmt.correctStatistics,
            self.model_dir,
            f"correctstatsepoch={epoch}.pt",
        )
        self.forget_msrmt.resetTrainIter()

    def train_epoch(self, ckpt, epoch):
        if ckpt is None:  # initialize model and return ckpt
            model, optimizer = self.get_model_optim()
            return self.checkpoint(
                model,
                optimizer,
                0.0,
                0.0,
            )

        # else train ckpt for 1 epoch
        model, optimizer = self.get_model_optim(
            ckpt["model_state_dict"], ckpt["optimizer_state_dict"]
        )
        batch_loss, batch_acc, eval_logits = [], [], []
        for iter, batch in enumerate(self.dataloader):
            t_0 = time.perf_counter()
            # metrics: output logits for all examples in eval dataset
            # do this before training to match Toneva
            model.eval()
            eval_logits.append(model(self.eval_x).detach().cpu())
            model.train()

            t_1 = time.perf_counter()
            x, y = batch
            logits = model(x.cuda())
            self.forget_msrmt.trackForgettableExamples(logits.detach(), y.detach())

            J = self.loss(logits, y.cuda())
            model.zero_grad()
            J.backward()
            optimizer.step()

            loss = J.item()
            acc = y.eq(logits.detach().argmax(dim=1).cpu()).float().mean()
            batch_loss.append(loss)
            batch_acc.append(acc)
            t_2 = time.perf_counter()
            self.forget_msrmt.incrementTrainBatch()
            print(
                f"\t{epoch}:{iter}\ta={acc:0.2f} l={loss:0.4f} ttime={t_2 - t_1:0.4f} mtime={t_1 - t_0:0.4f}"
            )

        # metrics: save per-iteration probabilities
        self.job.save_obj_to_subdir(
            torch.stack(eval_logits, axis=0), self.model_dir, f"eval_logits={epoch}.pt"
        )
        # WARNING: forget_msrmt is legacy code, will not work correctly if training from epoch > 0
        self.forget_msrmt.resetTrainBatchTracker()
        for logits_prime, y in self.evaluate_model(model, self.dataloader):
            self.forget_msrmt.trackCorrectExamples(logits_prime, y)
            self.forget_msrmt.incrementClassifyBatch()
        self.forget_msrmt.resetClassifyBatchTracker()
        self.forget_msrmt.incrementTrainIter()

        return self.checkpoint(
            model,
            optimizer,
            torch.tensor(batch_loss).mean().item(),
            torch.tensor(batch_acc).mean().item(),
        )

    def get_model_optim(self, model_state_dict=None, optim_state_dict=None):
        model = self.job.get_model(state_dict=model_state_dict)
        if self.job.hparams["model parameters"] == "default":
            optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        if optim_state_dict is not None:
            optimizer.load_state_dict(optim_state_dict)
        return model, optimizer

    def checkpoint(self, model, optimizer, loss, accuracy):
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
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
