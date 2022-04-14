import os
import time
import types
import datetime
from math import ceil
from pathlib import Path
import torch
import numpy as np
import matplotlib
from torchvision import datasets
import torchvision.transforms

from forget.metrics.transforms import stats_str


HPARAMS = {
    "CIFAR10": {
        "n_train_examples": 50000,
        "n_test_examples": 10000,
        "n_unique_labels": 10,
        "example_shape": (3, 32, 32),
    },
    "CIFAR100": {
        "n_train_examples": 50000,
        "n_test_examples": 10000,
        "n_unique_labels": 100,
        "example_shape": (3, 32, 32),
    },
}


class Job:
    def __init__(self, name, exp_path, data_dir, hparams):
        self.name = name
        self.hparams = hparams
        self.data_dir = data_dir
        self.save_path = os.path.join(exp_path, name)
        self.batch_size = int(self.hparams["batch size"])
        self.n_iter_per_epoch = ceil(self.n_train_examples / self.batch_size)
        self.augment_train_data = self.hparams["augment train data"] == "True"
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

    def file_exists(self, subdir, filename):
        file = os.path.join(self.save_path, subdir, filename)
        return os.path.exists(file)

    def cached(
        self, gen_fn, subdir, filename, overwrite=False, to_cpu=False, use_numpy=False
    ):
        dir = os.path.join(self.save_path, subdir)
        file = os.path.join(dir, filename)
        if use_numpy:
            file = file + ".npy"
        if not os.path.exists(file) or overwrite:
            start_time = time.perf_counter()
            print(f"Generating {subdir}/{filename} with {gen_fn.__name__}")
            obj = gen_fn()
            if type(obj) is np.ndarray:
                print(f"\t{stats_str(obj)} t={time.perf_counter() - start_time}")
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

    def get_model(self, state_dict=None, to_cuda=True):
        from open_lth.foundations import hparams
        from open_lth.models import registry

        model_type = self.hparams["model parameters"]
        assert (
            model_type == "cifar_resnet_20"
            or model_type == "cifar_resnet_14_64"
            or model_type == "cifar_resnet_110"
        )
        _model_params = hparams.ModelHparams(model_type, "kaiming_uniform", "uniform")
        model = registry.get(_model_params, outputs=self.n_unique_labels)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        if to_cuda:
            model = model.cuda()
        return model

    @property
    def n_train_examples(self):
        return HPARAMS[self.hparams["dataset"]]["n_train_examples"]

    @property
    def n_eval_examples(self):
        return self.n_logit_train_examples + self.n_logit_test_examples

    @property
    def n_unique_labels(self):
        return HPARAMS[self.hparams["dataset"]]["n_unique_labels"]

    @property
    def example_shape(self):
        return HPARAMS[self.hparams["dataset"]]["example_shape"]

    @property
    def n_logit_train_examples(self):
        return int(self.hparams["eval number of train examples"])

    @property
    def n_logit_test_examples(self):
        return int(self.hparams["eval number of test examples"])

    def _get_dataset(self, train=True, start=0, end=-1, augment_data=False):
        dataset_name = self.hparams["dataset"]
        if dataset_name == "CIFAR10" or dataset_name == "CIFAR100":
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )
            if augment_data:
                transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.RandomCrop(32, 4),
                        transform,
                    ]
                )
            if dataset_name == "CIFAR10":
                dataset = datasets.CIFAR10(
                    self.data_dir,
                    train=train,
                    download=False,
                    transform=transform,
                )
            elif dataset_name == "CIFAR100":
                dataset = datasets.CIFAR100(
                    self.data_dir,
                    train=train,
                    download=False,
                    transform=transform,
                )
        else:
            raise ValueError(f"'dataset'={dataset_name} is not defined.")
        if start == 0 and end <= 0:
            return dataset
        return torch.utils.data.Subset(dataset, torch.arange(start, end))

    def get_train_dataset(self, augment_data: bool = None):
        if augment_data is None:
            augment_data = self.augment_train_data
        return self._get_dataset(train=True, augment_data=self.augment_train_data)

    def get_eval_dataset(self):
        train = self._get_dataset(train=True, start=0, end=self.n_logit_train_examples)
        test = self._get_dataset(train=False, start=0, end=self.n_logit_test_examples)
        return torch.utils.data.ConcatDataset([train, test])

    def get_test_dataset(self):
        return self._get_dataset(train=False)

    def get_eval_labels(self):
        return np.array([y for _, y in self.get_eval_dataset()])

    def load_from_replicate(
        self, replicate: int, epoch: int, file_prefix, to_cpu=False
    ):
        epoch = [x for x in range(self.n_epochs + 1)][epoch]
        replicate = [x for x in range(self.n_replicates)][replicate]
        file = os.path.join(
            self.save_path, f"model{replicate}", f"{file_prefix}{epoch}.pt"
        )
        if to_cpu:
            checkpoint = torch.load(file, map_location=torch.device("cpu"))
        else:
            checkpoint = torch.load(file)
        return checkpoint

    def load_checkpoints_by_epoch(self, epoch=-1, to_cpu=False):
        for i, _ in enumerate(self.replicates()):
            yield self.load_from_replicate(
                i, epoch, file_prefix="ckpt-ep", to_cpu=to_cpu
            )

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
