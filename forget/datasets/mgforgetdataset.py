import torch

class manageForgetDataset:
    def __init__(self, forget_msrmt, forget_thres = 3):
        self.forget_stats = forget_msrmt.forgetStatistics
        self.correct_stats = forget_msrmt.correctStatistics
        self.sum_over_ep_flatten_forget = torch.flatten(torch.sum(self.forget_stats, 0))
        self.forget_thres = forget_thres
        self.trainset = datasets.CIFAR10('/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
        self.forget_mask = list()
        self.forget_mask_correct = list()
        self.batch_size = self.forget_stats[0,0].size()[0] #infer batch size from input
        self.forgotten_correct_stats = list()

    #get a mask of most forgotten examples
    def getForgetMask(self):
        for k in range(len(self.sum_over_ep_flatten_forget)):
            if self.sum_over_ep_flatten_forget[k] >= self.forget_thres:
                self.forget_mask.append(k)
        return self.forget_mask

    #get a mask of most forgotten examples that were classified *correctly* at the end of training
    def getForgetMaskCorrect(self, which_epoch = None):
        if which_epoch == None:
            at_epoch = len(self.correct_stats)
        else:
            at_epoch = which_epoch
    
        correct_flat = torch.flatten(self.correct_stats[at_epoch-1])

        #note that to add the which_epoch functionality correctly
        #I'll have to make sure sum_over_ep_flatten is only summed
        #up to which_epoch... can add that in later **TODO!!**
        #right now it just assumes we mean the last epoch of training

        for k in range(len(correct_flat)):
            if self.sum_over_ep_flatten_forget[k] >= self.forget_thres and correct_flat[k]==1:
                self.forget_mask_correct.append(k)
                self.forgotten_correct_stats.append(torch.IntTensor.item(self.sum_over_ep_flatten_forget[k]))

        return self.forget_mask_correct

    def get_forgotten_dataset_correct(self): #return a mask of those examples that were forgotten AND classified correctly
        #requires having run getForgetMask() first
        train_subset_correct = torch.utils.data.Subset(self.trainset, self.getForgetMaskCorrect())
        return torch.utils.data.DataLoader(train_subset_correct, batch_size=self.batch_size, num_workers = 0)

    def get_num_forgotten(self):
        if len(self.forget_mask)==0:
            raise ValueError("Obtain the mask of forgettable examples first.")
            
        return len(self.forget_mask)

    def get_num_forgotten_correct(self, which_epoch = None):
        if which_epoch == None:
            return len(self.forget_mask_correct)
        else:
            return len(self.getForgetMaskCorrect(which_epoch))

    def get_forgotten_dataset(self):
        train_subset = torch.utils.data.Subset(self.trainset, self.getForgetMask())
        return torch.utils.data.DataLoader(train_subset, batch_size=self.batch_size)