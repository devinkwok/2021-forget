import os
import torch
import datetime
from pathlib import Path
import matplotlib
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


class Job():

    def __init__(self, name, exp_path, data_dir, hparams):
        self.name = name
        self.hparams = hparams
        self.data_dir = data_dir
        self.save_path = os.path.join(exp_path, name)
        for subdir in self.replicate_dirs():
            Path(subdir).mkdir(parents=True, exist_ok=True)

    @property
    def n_replicates(self):
        return int(self.hparams['num replicates'])

    @property
    def n_epochs(self):
        return int(self.hparams['num epochs']) + 1

    def replicate_dirs(self):
        for i in range(self.n_replicates):
            yield os.path.join(self.save_path, f'model{i}')

    def save_obj_to_subdir(self, obj, subdir, filename):
        dir = os.path.join(self.save_path, subdir)
        file = os.path.join(dir, filename)
        Path(dir).mkdir(parents=True, exist_ok=True)
        print(f'Saving {file} to {dir}, t={datetime.datetime.now()}')
        if obj == matplotlib.pyplot:
            obj.savefig(file)
        else:
            torch.save(obj, file)
        return file

    def get_model(self, state_dict=None):
        from open_lth.foundations import hparams
        from open_lth.models import registry

        model_type = self.hparams["model parameters"] 
        if model_type == "default":
            _model_params = hparams.ModelHparams(
                'cifar_resnet_20',
                'kaiming_uniform',
                'uniform'
            )
            model = registry.get(_model_params).cuda()
        else:
            raise ValueError(f"'model parameters'={model_type} is not defined")
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return model

    def _get_dataset(self, train=True, start=0, end=-1):
        dataset_name = self.hparams['dataset']
        if dataset_name == 'CIFAR10':
            dataset = datasets.CIFAR10(
                self.data_dir,
                train=train,
                download=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )])
            )
        else:
            raise ValueError(f"'dataset'={dataset_name} is not defined.")
        if start == 0 and end <= 0:
            return dataset
        return torch.utils.data.Subset(dataset, torch.arange(start, end))

    def get_dataloader(self, train=True):
        if train:
            dataset = self.get_train_dataset()
            batch_size = int(self.hparams["batch size"])
        else:
            dataset = self.get_eval_dataset()
            batch_size = int(self.hparams["eval batch size"])
        return DataLoader(dataset, batch_size=batch_size, num_workers=0)

    def get_train_dataset(self):
        return self._get_dataset(train=True)

    def get_eval_dataset(self):
        train = self._get_dataset(train=True, start=0,
                end=int(self.hparams["eval number of train examples"]))
        test = self._get_dataset(train=False, start=0,
                end=int(self.hparams["eval number of test examples"]))
        return torch.utils.data.ConcatDataset([train, test])

    #TODO this is not an efficient abstraction due to loading from other dirs (e.g. noise checkpoints)
    def load_checkpoints(self, epoch_idx=-1, to_cpu=False):
        # use list indexing to comprehend idx (+1 includes init at epoch 0)
        epochs = [x for x in range(self.n_epochs)]
        epoch = epochs[epoch_idx]
        for dir in self.replicate_dirs():
            fname = os.path.join(dir, f'epoch={epoch}.pt')
            if to_cpu:
                model = torch.load(fname, map_location=torch.device('cpu'))
            else:
                model = torch.load(fname)
            yield model
