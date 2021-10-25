import os
import time
import torch
import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms, utils
from torch.utils.data import random_split, DataLoader
from pathlib import Path
from forget.training import measureforget


class train:

    def __init__(self, model, job, replicate_num):
        #structure of directory is eg ../jobs/job1/model1/
        self.job = job
        self.replicate_num = replicate_num
        self.save_every = int(self.job.hparams["save every"])
        self.model_dir = f"model{self.replicate_num}"
        self.model = model

        if self.job.hparams["model parameters"] == "default":
            self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)
            self.loss = nn.CrossEntropyLoss()
        else:
            pass #to add this functionality, custom loss and optimizer

        if bool(self.job.hparams["save models"]):
            self.save_model = True
        else:
            self.save_model = False
        self.batch_size = int(self.job.hparams['batch size']) #note that this also gets passed to measureForget

        self.dataloader = self.job.get_dataloader(train=True)
        # metrics: create eval dataset
        self.eval_dataloader = self.job.get_dataloader(train=False)
        self.eval_x = [x.cuda() for x, y in self.eval_dataloader]

        self.forget_msrmt = measureforget.measureForget(self.job.n_epochs, num_batches = len(self.dataloader), batch_size=self.batch_size)


    def trainLoop(self):
        # save checkpoint at initialization
        self.save_checkpoint(0, 0, 0)

        for epoch in range(self.job.n_epochs):
            batch_loss, batch_acc, eval_logits = [], [], []
            self.model.train()
            for iter, batch in enumerate(self.dataloader):
                t_0 = time.perf_counter()
                # metrics: output logits for all examples in eval dataset
                # do this before training to match Toneva
                eval_logits.append(torch.cat(
                    [x.cpu() for x, _ in self.evaluate_model(self.eval_dataloader)],
                    dim=0))

                t_1 = time.perf_counter()
                x, y = batch
                x = x.cuda()
                logits = self.model(x)
                self.forget_msrmt.trackForgettableExamples(logits.detach(), y.detach())

                J = self.loss(logits, y.cuda())
                self.model.zero_grad()
                J.backward()
                self.optimizer.step()

                loss = J.item()
                acc = y.eq(logits.detach().argmax(dim=1).cpu()).float().mean()
                batch_loss.append(loss)
                batch_acc.append(acc)
                t_2 = time.perf_counter()
                self.forget_msrmt.incrementTrainBatch()
                print(f'Ep{epoch} it{iter} l={loss:0.4f} a={acc:0.2f} ttime={t_2 - t_1:0.4f} mtime={t_1 - t_0:0.4f}')
            
            self.forget_msrmt.resetTrainBatchTracker()
            for logits_prime, y in self.evaluate_model(self.dataloader):
                self.forget_msrmt.trackCorrectExamples(logits_prime, y)
                self.forget_msrmt.incrementClassifyBatch()
            self.forget_msrmt.resetClassifyBatchTracker()

            if (epoch+1) % self.save_every == 0:
                self.save_checkpoint(epoch + 1,
                    torch.tensor(batch_loss).mean().item(),
                    torch.tensor(batch_acc).mean().item())
                self.job.save_obj_to_subdir(self.forget_msrmt.forgetStatistics,
                    self.model_dir, f"forgetstatsepoch={epoch + 1}.pt")
                self.job.save_obj_to_subdir(self.forget_msrmt.correctStatistics,
                    self.model_dir, f"correctstatsepoch={epoch + 1}.pt")
                # metrics: save per-iteration probabilities
                self.job.save_obj_to_subdir(torch.stack(eval_logits, axis=0),
                    self.model_dir, f"eval_logits={epoch + 1}.pt")
            self.forget_msrmt.incrementTrainIter()

        self.forget_msrmt.resetTrainIter()
        self.clean()

    def evaluate_model(self, dataloader):
        self.model.eval()
        for x, y in dataloader:
            x = x.cuda()
            with torch.no_grad():
                output = self.model(x)
            yield output.detach(), y.detach()
        self.model.train()

    def save_checkpoint(self, epoch, loss, accuracy):
        ckpt = {
           'epoch': epoch,
           'model_state_dict': self.model.state_dict(),
           'optimizer_state_dict': self.optimizer.state_dict(),
           'loss': loss,
           'train accuracy': accuracy,
           }
        self.job.save_obj_to_subdir(ckpt, self.model_dir, f"epoch={epoch}.pt")

    def clean(self):
        # after training, clean caches,..
        del self.model
        del self.forget_msrmt
