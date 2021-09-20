"""
Class for managing forget data. It can go through forget statistics and output
a mask corresponding to the forget threshold. This mask can be used e.g. to obtain a
dataset of forgettable examples. It can also output subset datasets of correctly
classified examples and forgotten + correctly classified examples at the end of
training.
"""

import torch
import os
import sys
#sys.path.append(str("/")) #this will need to be fixed later
from forget.config import parser
from forget.training import trainer
import os
from pathlib import Path

class createForgetDataset:
    def __init__(self, forget_thres = 3, config_file = os.getcwd()+"/Forget/config/default_config.ini"):
        parent_dir_path = Path(Path().absolute()).parent #os.path.dirname(str(os.path.dirname(os.path.realpath(__file__))))
        print(f"Appending path: {str(parent_dir_path)}")
        sys.path.append(str(parent_dir_path)+"/")
        #assume latest epoch, go through forgetstats and correctstats
        #for each model, initialize the below variables, create the forget
        #dataset + save into forgetdata/ and also the forget+correct dataset.

        #first set the threshold
        self.forget_thres = forget_thres

        #from the config file, obtain experiment name + number of jobs + number of models
        self.reader = parser.readConfig(config_file)

        self.exp_name = self.reader.exp_info["name"]
        #self.num_jobs = int(self.reader.exp_info["number of jobs"])
        
        self.list_forget_folders = []
        print(f"Reading from output files in experiment {self.exp_name}...")
        for job in self.reader.jobs: #remember to put everything into forgetdata
            mypathdbg = str(parent_dir_path) + "/"+self.exp_name + "/" + job
            print(f"Looking for subdirectories in : {mypathdbg}")
            job_subdir = [str(parent_dir_path) + "/"+self.exp_name + "/" + job + "/" + f.name + "/forgetdata/" for f in os.scandir(str(parent_dir_path) + "/"+self.exp_name + "/" + job + "/") if f.is_dir()]
            for dir in job_subdir:
                self.list_forget_folders.append(dir)
            #self.list_correct_folders = [self.exp_name + "/" + job + "/" + f.name + "/correctdata/" for f in os.scandir("/" + self.exp_name + "/" + job + "/") if f.is_dir()]
            self.max_epoch = int(self.reader.jobs[job]["num epochs"])

        """
        should move dataset info to experiment probably
        so this part needs to be systematized better.
        right now we just assume that the dataset is CIFAR10
        but soon we'll want to do other datasets and we should
        just write a class that can fetch a dataset from exp_info.
        Perhaps I should dispense with the "job" system entirely?
        """
        print(f"Fetching dataset...")
        self.trainset = fetch_dataset(data_idx = 0, batch_size = 128)
        
        """
        for each model go through forgetstats and correctstats, upload the masks to the folder.
        This part is undoubtedly clunky... ideally we'd have a directory just associated with the model
        and scan over models, rather than scan over directories.
        Maybe something to improve in the next iteration of this code.
        """
        print(self.list_forget_folders)

        self.getForgetMask_has_run = False
        self.getForgetMaskCorrect_has_run = False
        
        for folder_idx, subfolder in enumerate(self.list_forget_folders):
            self.forget_stats = torch.load(subfolder + "forgetstatsepoch="+str(self.max_epoch)+".pt")
            self.correct_stats = torch.load(subfolder + "correctstatsepoch="+str(self.max_epoch)+".pt")
            self.sum_over_ep_flatten_forget = torch.flatten(torch.sum(self.forget_stats, 0))
            self.forget_mask = list()
            self.forget_mask_correct = list()
            self.batch_size = self.forget_stats[0,0].size()[0] #infer batch size from input
            self.forgotten_correct_stats = list()

            #now compute the masks and save them
            self.getForgetMask(save_directory = subfolder)
            self.getForgetMaskCorrect(save_directory = subfolder)

            #obtain the forgotten dataset that's classified correctly and save it
            self.getFgtDatasetCorrect(save_directory = subfolder)

            #as well as the full forgotten dataset
            self.getFullForgottenDataset(save_directory = subfolder)

            print(f"Number forgotten: {self.get_num_forgotten()}")
            print(f"Number forgotten + correct: {self.get_num_forgotten_correct()}")
            torch.save(torch.tensor([self.get_num_forgotten(), self.get_num_forgotten_correct()]), subfolder + "num_forgotten.pt")


    def getForgetMask(self, save_directory = "/", save = True):
        self.getForgetMask_has_run = True
        for k in range(len(self.sum_over_ep_flatten_forget)):
            if self.sum_over_ep_flatten_forget[k] >= self.forget_thres:
                self.forget_mask.append(k)
        if save:
            torch.save(torch.tensor(self.forget_mask), save_directory + "forgetmask_epoch=" + str(self.max_epoch) + ".pt")

    def getForgetMaskCorrect(self, save_directory = "/", save = True):
        self.getForgetMaskCorrect_has_run = True
        self.correct_flat = torch.flatten(self.correct_stats[self.max_epoch-1])

        for k in range(len(self.correct_flat)):
            if self.sum_over_ep_flatten_forget[k] >= self.forget_thres and self.correct_flat[k]==1:
                self.forget_mask_correct.append(k)
                self.forgotten_correct_stats.append(torch.IntTensor.item(self.sum_over_ep_flatten_forget[k]))

        if save:
            torch.save(torch.tensor(self.forgotten_correct_stats), save_directory + "forgotten_correct_stats_epoch=" + str(self.max_epoch) + ".pt")
            torch.save(torch.tensor(self.forget_mask_correct), save_directory + "forgetmask_correct_epoch=" + str(self.max_epoch) + ".pt")
    
    def getFgtDatasetCorrect(self, save_directory = "/", save = True): #return a mask of those examples that were forgotten AND classified correctly
        #requires having run getForgetMask() first
        if not self.getForgetMaskCorrect_has_run:
            raise ValueError("Run getForgetMaskCorrect() first!")

        self.train_subset_correct = torch.utils.data.Subset(self.trainset, self.forget_mask_correct)

        if save:
            torch.save(torch.utils.data.DataLoader(self.train_subset_correct, batch_size=self.batch_size, num_workers = 0),
                       save_directory + "forgotten_correct_dataloader_epoch=" + str(self.max_epoch) + ".pt")

    def get_num_forgotten(self):
        if len(self.forget_mask)==0:
            raise ValueError("Obtain the mask of forgettable examples first; right now it's empty.")
        else:
            return len(self.forget_mask)

    def get_num_forgotten_correct(self, which_epoch = None):
        if len(self.forget_mask_correct) == 0:
            raise ValueError("Obtain mask of forgettable + correct examples first; right now it's empty.")
        else:
            return len(self.forget_mask_correct)

    def getFullForgottenDataset(self, save_directory = "/", save = True):
        if not self.getForgetMask_has_run:
            raise ValueError("Run getForgetMask() first!")
        self.train_subset = torch.utils.data.Subset(self.trainset, self.forget_mask)
        self.full_forgotten_dataset = torch.utils.data.DataLoader(self.train_subset, batch_size=self.batch_size)

        if save:
            torch.save(self.full_forgotten_dataset,
                       save_directory + "full_forgotten_dataloader_epoch=" + str(self.max_epoch) + ".pt")

from torchvision import datasets, transforms, utils
from torch.utils.data import random_split, DataLoader

def fetch_dataset(data_idx, batch_size):
    if data_idx == 0:
        train_dataset = datasets.CIFAR10(os.getcwd(), train=True, download=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
        return DataLoader(train_dataset, batch_size, num_workers = 0)