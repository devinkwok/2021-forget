import torch
import os, os.path
import sys
from open_lth.foundations import hparams
from open_lth.models import registry
from forget.main import parser
import numpy as np
from torchvision import datasets, transforms, utils
from torch.utils.data import random_split, DataLoader
from pathlib import Path


class postProcess:
    """
    Go through clones/ and for each clone load the forget_correct dataset and classify it.
    """
    def __init__(self, config_file = os.getcwd()+"/Forget/config/default_config.ini"):
        self.reader = parser.readConfig(config_file)
        self.exp_name = self.reader.exp_info["name"]
        parent_dir_path = Path(Path().absolute()).parent
        #self.num_jobs = int(self.reader.exp_info["number of jobs"])
        
        self.list_clone_folders = []
        self.list_model_folders = []
        self.num_examples = []
        self.num_forgotten_correct = []
        self.model_counts = []
        print(f"Reading from clones files in experiment {self.exp_name}...")
        for job in self.reader.jobs: #remember to put everything into forgetdata
            clone_subdir = [str(parent_dir_path)+"/"+self.exp_name + "/" + job + "/" + f.name + "/clones/" for f in os.scandir(str(parent_dir_path) + "/" + self.exp_name + "/" + job + "/") if f.is_dir()]
            model_subdir = [str(parent_dir_path)+"/"+self.exp_name + "/" + job + "/" + f.name for f in os.scandir(str(parent_dir_path) + "/" + self.exp_name + "/" + job + "/") if f.is_dir()]
            for dir in model_subdir:
                self.num_examples.append(torch.load(dir + "/forgetdata/num_forgotten.pt")[1])
                self.list_model_folders.append(dir)
            
            for dir in clone_subdir:
                self.list_clone_folders.append(dir)
                self.model_counts.append(len(os.listdir(dir)))

            if self.reader.jobs[job]["model parameters"] == "default":
                self.model_hparams = hparams.ModelHparams('cifar_resnet_20', 'kaiming_uniform', 'uniform')
            else:
                pass #to add in
            self.max_epoch = int(self.reader.jobs[job]["num epochs"])
        
        print(f"Model counts: {self.model_counts}")
        print(f"Clone folders: {self.list_clone_folders}")
        #scan and find models

        self.epsilons = getEpsilons()
        self.totalEpsilons = list()
        self.totalForgotten = list()
        
        for clone_idx, clone_dir in enumerate(self.list_clone_folders):
            self.train_set = datasets.CIFAR10(os.getcwd(), train=True, download=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])) # torch.load(self.list_model_folders[clone_idx] + "/forgetdata/trainset.pt")
            self.forgot_correct_mask = torch.load(self.list_model_folders[clone_idx] + "/forgetdata/forgetmask_correct_epoch=" + str(self.max_epoch) + ".pt")
            self.forgot_correct_dataset = torch.utils.data.Subset(self.train_set, self.forgot_correct_mask)
            self.forget_correct_dataloader = torch.utils.data.DataLoader(self.forgot_correct_dataset, batch_size = 128) #to change: this call to batch_size
            self.model_list = []
            
            for idx in range(self.model_counts[clone_idx]):
                self.model = registry.get(self.model_hparams).cuda()
                self.model.load_state_dict(torch.load(clone_dir+str(idx)+".pt"))
                self.model_list.append(self.model)

            catalog = self.classifyDataset(self.forget_correct_dataloader, self.model_list, self.num_examples[clone_idx])
            torch.save(catalog, self.list_model_folders[clone_idx]+"/catalog.pt")

            self.forget_correct_stats = torch.load(self.list_model_folders[clone_idx] + "/forgetdata/forgotten_correct_stats_epoch=" + str(self.max_epoch) + ".pt")
            self.epsilonForgotten, self.timesForgotten = self.tabulateNoiseForget(catalog, self.epsilons, self.forget_correct_stats)
            self.totalEpsilons.append(self.epsilonForgotten)
            self.totalForgotten.append(self.timesForgotten)
            torch.save(torch.tensor(self.epsilonForgotten),  self.list_model_folders[clone_idx]+"/epsilonForgotten.pt")
            torch.save(torch.tensor(self.timesForgotten),  self.list_model_folders[clone_idx]+"/timesForgotten.pt")

            #print(self.totalEpsilons)

            self.totalEpsilonsTensor = torch.flatten(torch.Tensor([item for sublist in self.totalEpsilons for item in sublist]))
            self.totalForgottenTensor = torch.flatten(torch.Tensor([item for sublist in self.totalForgotten for item in sublist]))

            torch.save(self.totalEpsilonsTensor, self.list_model_folders[clone_idx]+"/epsilontotal.pt")
            torch.save(self.totalForgottenTensor, self.list_model_folders[clone_idx]+"/timesforgottentotal.pt")

    #measure at which noise level an example that's classified correctly becomes misclassifed
    #this function just classifies a dataset given a model
    def classifyDataset(self, data_loader, models, num_examples):
        if self.num_examples==None:
            raise ValueError("Specify the size of the dataset please.")

        num_models = len(models)
        __catalog = torch.zeros(num_models, num_examples)

        num_ex_per_batch = list()
        for batch in data_loader:
            num_ex_per_batch.append(len(batch[0]))
        print(f"Classifying dataset... examples/batch: {num_ex_per_batch}")
        
        modeltrcker = 0
        for model in models:
            model.eval()

            btrkcer = 0
            for batch in data_loader:
                x,y = batch
                x=x.cuda()
                with torch.no_grad():
                    l_A = model(x)
                for k in range(len(l_A)):
                    if torch.argmax(l_A[k]) == y.cuda()[k]:
                        #print(f"{modeltrcker}, {k+sum(num_ex_per_batch[0:btrkcer])}")
                        __catalog[modeltrcker, k+sum(num_ex_per_batch[0:btrkcer])] = 1
                btrkcer+=1        
                
            modeltrcker+=1

        return __catalog
    
    #returns a table consisting of {epsilon at which example was forgotten, times it was forgotten}
    def tabulateNoiseForget(self, catalog, epsilonList, forgetCorrectStats):
        epsilonForgotten = list()
        timesForgotten = list()

        for k in range(len(catalog[0])): #go through each example
            idx = next((i for i in range(len(catalog[0:,k])) if catalog[0:,k][i] == 0), None)
            if idx != None:
                epsilonForgotten.append(epsilonList[idx])
                timesForgotten.append(forgetCorrectStats[k])
        
        return epsilonForgotten, timesForgotten

    #For a given set of epsilon, forget stats, it scans through and determines the
    #largest N epsilons for a given # of forgotten events before the example was
    #misclassifed
    def findLargestEpsilon(self, epsilonForgotten, timesForgotten, largestN):
        import heapq
        largest_value = int(max(timesForgotten))
        smallest_value = int(min(timesForgotten))

        largest_epsilon = torch.zeros(largest_value-smallest_value+1, largestN)
        largest_forgotten = torch.zeros(largest_value-smallest_value+1, largestN)

        for j in range(smallest_value, largest_value+1):
            idx = [i for i in range(len(timesForgotten)) if timesForgotten[i]==j]
            for k in range(len(timesForgotten)):
                largest_forgotten[j-smallest_value, 0:] = torch.tensor([j]*largestN)
                largest_epsilon[j-smallest_value, 0:] = torch.tensor(heapq.nlargest(largestN,[epsilonForgotten[i] for i in idx]))

        return torch.flatten(largest_epsilon), torch.flatten(largest_forgotten)

def getEpsilons(num_points = 200, min_noise = 0., max_noise = 0.1):
        return np.linspace(min_noise, max_noise, num_points)