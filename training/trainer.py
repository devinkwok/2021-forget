import torch
from torch import nn, optim
from torchvision import datasets, transforms, utils
from torch.utils.data import random_split, DataLoader
from pathlib import Path
from Forget.training  import measureforget
from Forget.training  import metrics
import os

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
        
        self.dataloader = self.getDataset(job_info["dataset"], self.batch_size, train=True)

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
        self.top_k_class, self.top_k_score = [], []
        # metrics: create eval dataset (TODO: for now, just put all test examples)
        self.eval_dataloader = self.getDataset(job_info["dataset"], self.batch_size * 8, train=False)

    def getDataset(self, dataset_name, batch_size, train=True): #option to change batch size?
        print(f"Loading train dataset {dataset_name}... batch size {batch_size}")
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
            return DataLoader(dataset, batch_size=batch_size, num_workers=0)
        raise ValueError(f'Dataset {dataset_name} not found.')

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
            for batch in self.dataloader:
                x,y = batch
                x=x.cuda()
                logits = model(x)

                if self.forget_flag: #eventually should change forget class to have wrapper instead of these flags.
                    self.forget_msrmt.trackForgettableExamples(logits.detach(), y.detach())

                J = self.loss(logits, y.cuda())
                model.zero_grad()
                J.backward()
                self.optimizer.step()

                batch_loss.append(J.item())
                batch_acc.append(y.eq(logits.detach().argmax(dim=1).cpu()).float().mean())

                # metrics: generate per-iteration metrics (top_k classes and probabilities)
                top_k_outputs = [metrics.top_k(x, y)
                    for x, y in self.evaluate_model(
                        model, self.eval_dataloader, return_probabilities=True)]
                # invert nesting order of list from [[class_1, score_1], [class_2, score_2], ...]
                # to [[class_1, class_2, ...], [score_1, score_2, ...]], then combine into ndarray
                top_k_class, top_k_score = [np.concatenate(x, axis=0) for x in zip(*top_k_outputs)]
                self.top_k_class.append(top_k_class)
                self.top_k_score.append(top_k_score)

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
                self.save_data()
            
            accuracies.append(torch.tensor(batch_acc).mean())
            if self.forget_flag:
                self.forget_msrmt.incrementTrainIter()

        if self.forget_flag:
            self.forget_msrmt.resetTrainIter()
        
        model.eval()
        self.clean(model)

    def evaluate_model(self, model, dataloader, return_probabilities=False, as_numpy=False):
        model.eval()
        for x, y in dataloader:
            x = x.cuda()
            with torch.no_grad():
                output = model(x)
                if return_probabilities:
                    output = torch.softmax(output, dim=1)
            output = output.detach()
            y = y.detach()
            if as_numpy:
                output = output.numpy()
                y = y.numpy()
            yield output, y
        model.train()

    def save_model_data(self, model, epoch, loss, accuracy):
        torch.save({
           'epoch': epoch+1,
           'model_state_dict': model.state_dict(),
           'optimizer_state_dict': self.optimizer.state_dict(),
           'loss': loss,
           'train accuracy': accuracy,
           }, self.store_directory + "epoch=" + str(epoch+1) + ".pt")
        
    def save_data(self):
        if self.forget_flag:
            self.forget_msrmt.saveForget(self.store_directory)
        if self.track_correct_ex:
            self.forget_msrmt.saveCorrect(self.store_directory)

        #to add: save accuracies

        # metrics: save per-iteration classes and probabilities
        torch.save(self.top_k_class, self.store_directory + "metrics/" + "top_k_class.pt")
        torch.save(self.top_k_score, self.store_directory + "metrics/" + "top_k_score.pt")

    def clean(self, model):
        del model
        del self.forget_msrmt

        #after training, clean caches,..
