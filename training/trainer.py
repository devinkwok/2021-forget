import os
import time
import torch
import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms, utils
from torch.utils.data import random_split, DataLoader
from pathlib import Path
from Forget.training  import measureforget
from Forget.training  import metrics

class train:
    def __init__(self, model, exp_info, job_info, job_idx, model_idx): #job_idx, model_idx should be a unique modifier that indexes the job, model
        #structure of directory is eg ../jobs/job1/model1/
        #idx here would be '1'

        #list of datasets that trainer knows about
        parent_dir_path = Path(Path().absolute()).parent

        self.dataset_names = ['cifar10']
        self.num_epochs = int(job_info["num epochs"])
        self.save_every = int(job_info["save every"])
        if exp_info["storage directory"] == "default":
            self.exp_directory = str(parent_dir_path) + "/" + exp_info["name"] + "/"
        else:
            self.exp_directory = exp_info["storage directory"]

        if job_info["model parameters"] == "default":
            self.optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
            self.loss = nn.CrossEntropyLoss()
        else:
            pass #to add this functionality, custom loss and optimizer

        if job_info["save models"] == "true" or job_info["save models"] == "True":
            self.save_model = True
        else:
            self.save_model = False

        #print(job_info["dataset params"])
        if job_info["dataset params"] == "default":
            self.batch_size = 128 #note that this also gets passed to measureForget
        else:
            pass #to add, custom dataset batch size, num workers, etc.
        
        self.dataloader = self.get_dataloader(
            self.get_dataset(job_info["dataset"], train=True),
            self.batch_size)

        if job_info["measure forget"] == "true" or job_info["measure forget"] == "True":
            self.forget_flag = True
            self.forget_msrmt = measureforget.measureForget(self.num_epochs, num_batches = len(self.dataloader), batch_size=self.batch_size)
        else:
            self.forget_msrmt = None

        if job_info["track correct examples"] == "true" or job_info["track correct examples"] == "True":
            self.track_correct_ex = True
        
        if job_info["storage directory"] == "default":
            self.store_directory = self.exp_directory + "Job " + str(job_idx+1) + "/" + "model" + str(model_idx) + "/"
        else:
            pass #to add..
        
        #self.trainLoop(model) #train the model

        # metrics: initialize storage
        self.eval_logits = []
        # metrics: create eval dataset
        self.eval_dataloader = self.get_eval_dataloader(job_info["dataset"])

    def get_eval_dataloader(self, dataset_name):
        train_dataset = self.get_dataset(dataset_name, train=True, max_idx=5000)
        test_dataset = self.get_dataset(dataset_name, train=False, max_idx=5000)
        return self.get_dataloader(
            torch.utils.data.ConcatDataset([train_dataset, test_dataset]),
            batch_size=2500)

    def get_dataset(self, dataset_name, train=True, max_idx=-1):
        if dataset_name == 'CIFAR10':
            dataset = datasets.CIFAR10(
                os.getcwd(),  #TODO use localscratch location
                train=train,
                download=True,  #TODO change to False once localscratch is figured out
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )])
            )
        else:
            raise ValueError(f'Dataset {dataset_name} not found.')
        if max_idx > 0:  # truncate dataset to max_idx
            return torch.utils.data.Subset(dataset, np.arange(max_idx))
        return dataset

    def get_dataloader(self, dataset, batch_size: int):
        return DataLoader(dataset, batch_size=batch_size, num_workers=0)

    def trainLoop(self, model):
        losses = list()
        accuracies = list()
        epochs = list()

        # save checkpoint at initialization
        self.save_model_data(model, 0, 0, 0)

        for epoch in range(self.num_epochs):
            batch_loss = list()
            batch_acc = list()

            model.train()
            for iter, batch in enumerate(self.dataloader):
                t_start = time.perf_counter()

                x, y = batch
                x = x.cuda()
                logits = model(x)

                if self.forget_flag: #eventually should change forget class to have wrapper instead of these flags.
                    self.forget_msrmt.trackForgettableExamples(logits.detach(), y.detach())

                J = self.loss(logits, y.cuda())
                model.zero_grad()
                J.backward()
                self.optimizer.step()

                loss = J.item()
                acc = y.eq(logits.detach().argmax(dim=1).cpu()).float().mean()
                batch_loss.append(loss)
                batch_acc.append(acc)
                t_train = time.perf_counter()

                # metrics: output logits for all examples in eval dataset
                self.eval_logits.append(torch.cat(
                    [x.cpu() for x, _ in self.evaluate_model(model, self.eval_dataloader)],
                    dim=0))
                t_metrics = time.perf_counter()
                print(f'Ep{epoch} it{iter} l={loss} a={acc} ttime={t_train - t_start} mtime={t_metrics - t_train}')

                if self.forget_flag:
                    self.forget_msrmt.incrementTrainBatch()
            
            if self.forget_flag:
                self.forget_msrmt.resetTrainBatchTracker()

            if self.track_correct_ex:
                for logits_prime, y in self.evaluate_model(model, self.dataloader):
                    self.forget_msrmt.trackCorrectExamples(logits_prime, y)
                    self.forget_msrmt.incrementClassifyBatch()
    
                self.forget_msrmt.resetClassifyBatchTracker()

            if (epoch+1) % self.save_every == 0:
                self.save_model_data(model, epoch+1, torch.tensor(batch_loss).mean(), torch.tensor(batch_acc).mean())
                self.save_data(epoch+1)
            
            accuracies.append(torch.tensor(batch_acc).mean())
            if self.forget_flag:
                self.forget_msrmt.incrementTrainIter()

        if self.forget_flag:
            self.forget_msrmt.resetTrainIter()
        
        model.eval()
        self.clean(model)

    def evaluate_model(self, model, dataloader):
        model.eval()
        for x, y in dataloader:
            x = x.cuda()
            with torch.no_grad():
                output = model(x)
            yield output.detach(), y.detach()
        model.train()

    def save_model_data(self, model, epoch, loss, accuracy):
        save_location = self.store_directory + "epoch=" + str(epoch) + ".pt"
        print(f'Saving checkpoint to {save_location}')
        torch.save({
           'epoch': epoch,
           'model_state_dict': model.state_dict(),
           'optimizer_state_dict': self.optimizer.state_dict(),
           'loss': loss,
           'train accuracy': accuracy,
           }, save_location)
        
    def save_data(self, epoch: int):
        print(f'Saving data to {self.store_directory}')
        if self.forget_flag:
            self.forget_msrmt.saveForget(self.store_directory)
        if self.track_correct_ex:
            self.forget_msrmt.saveCorrect(self.store_directory)

        #to add: save accuracies

        # metrics: save per-iteration probabilities
        torch.save(torch.stack(self.eval_logits, axis=0),
            os.path.join(self.store_directory, f"eval_logits={epoch}.pt"))
        self.eval_logits = []  # clear memory

    def clean(self, model):
        del model
        del self.forget_msrmt

        #after training, clean caches,..
