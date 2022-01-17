import os
import types
import datetime
from math import ceil
from pathlib import Path
import torch
import numpy as np
import matplotlib
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


class Job:
    def __init__(self, name, exp_path, data_dir, hparams):
        self.name = name
        self.hparams = hparams
        self.data_dir = data_dir
        self.save_path = os.path.join(exp_path, name)
        self.batch_size = int(self.hparams["batch size"])
        self.n_iter_per_epoch = ceil(len(self.get_train_dataset()) / self.batch_size)
        self.n_train_examples = len(self.get_train_dataset())
        self.n_eval_examples = len(self.get_eval_dataset())
        for subdir, _ in self.replicates():
            Path(subdir).mkdir(parents=True, exist_ok=True)

    @property
    def n_replicates(self):
        return int(self.hparams["num replicates"])

    @property
    def n_epochs(self):
        return int(self.hparams["num epochs"])

    def replicates(self):
        for i in range(self.n_replicates):
            name = f"model{i}"
            yield os.path.join(self.save_path, name), name

    def cached(
        self, gen_fn, subdir, filename, overwrite=False, to_cpu=False, use_numpy=False
    ):
        dir = os.path.join(self.save_path, subdir)
        file = os.path.join(dir, filename)
        if use_numpy:
            file = file + ".npy"
        if not os.path.exists(file) or overwrite:
            print(f"Generating {subdir}/{filename} with {gen_fn.__name__}")
            obj = gen_fn()
            self.save_obj_to_subdir(obj, subdir, filename, use_numpy=use_numpy)
            del obj
        print(f"\tloading {subdir}/{filename}")
        if use_numpy:
            return np.load(file)
        elif to_cpu:
            return torch.load(file, map_location=torch.device("cpu"))
        else:
            return torch.load(file)

    def save_obj_to_subdir(self, obj, subdir, filename, use_numpy=False):
        dir = os.path.join(self.save_path, subdir)
        file = os.path.join(dir, filename)
        Path(dir).mkdir(parents=True, exist_ok=True)
        if type(obj) == types.ModuleType and obj == matplotlib.pyplot:
            obj.savefig(file)
            obj.close("all")
        elif use_numpy:
            np.save(file, obj, allow_pickle=False)
        else:
            torch.save(obj, file)
        print(f"Saved {filename} to {subdir}, t={datetime.datetime.now()}")
        return file

    def get_model(self, state_dict=None):
        from open_lth.foundations import hparams
        from open_lth.models import registry

        model_type = self.hparams["model parameters"]
        if model_type == "default" or model_type == "resnet20":
            _model_params = hparams.ModelHparams(
                "cifar_resnet_20", "kaiming_uniform", "uniform"
            )
            model = registry.get(_model_params)
        else:
            raise ValueError(f"'model parameters'={model_type} is not defined")
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return model.cuda()

    def _get_dataset(self, train=True, start=0, end=-1):
        dataset_name = self.hparams["dataset"]
        if dataset_name == "CIFAR10":
            dataset = datasets.CIFAR10(
                self.data_dir,
                train=train,
                download=False,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                        ),
                    ]
                ),
            )
        else:
            raise ValueError(f"'dataset'={dataset_name} is not defined.")
        if start == 0 and end <= 0:
            return dataset
        return torch.utils.data.Subset(dataset, torch.arange(start, end))

    def get_train_dataset(self):
        return self._get_dataset(train=True)

    def get_eval_dataset(self):
        train = self._get_dataset(
            train=True, start=0, end=int(self.hparams["eval number of train examples"])
        )
        test = self._get_dataset(
            train=False, start=0, end=int(self.hparams["eval number of test examples"])
        )
        return torch.utils.data.ConcatDataset([train, test])

    # TODO this is not an efficient abstraction due to loading from other dirs (e.g. noise checkpoints)
    def load_checkpoints_by_epoch(self, epoch_idx=-1, to_cpu=False):
        # use list indexing to comprehend idx (+1 includes init at epoch 0)
        epochs = [x for x in range(self.n_epochs + 1)]
        epoch = epochs[epoch_idx]
        for dir, name in self.replicates():
            file = os.path.join(dir, f"epoch={epoch}.pt")
            if to_cpu:
                model = torch.load(file, map_location=torch.device("cpu"))
            else:
                model = torch.load(file)
            yield model, name

    def load_checkpoints_from_dir(self, subdir, to_cpu=False):
        root = os.path.join(self.save_path, subdir)
        for file in os.listdir(root):
            name, suffix = os.path.splitext(file)
            if suffix == ".pt":
                fname = os.path.join(root, file)
                if to_cpu:
                    model = torch.load(fname, map_location=torch.device("cpu"))
                else:
                    model = torch.load(fname)
                yield model, name


def evaluate_one_batch(model, examples, labels):
    with torch.no_grad():
        output = model(examples).detach()
        accuracy = (
            torch.sum(torch.argmax(output, dim=1) == labels).float() / labels.shape[0]
        )
    return output, accuracy.item()
