import os
import typing
import numpy as np
import torch
from forget.postprocess import transforms


class Metrics:
    def __init__(self, job, plotter):
        # filter batch by train/test examples
        self.job = job
        self.subdir = f"metrics-ep{self.job.n_epochs}"
        self.plotter = plotter
        self.it_split = (
            int(self.job.hparams["plot early late epoch split point"])
            * self.job.n_iter_per_epoch
        )

    def gen_metrics(
        self,
        name: str,
        input_source: typing.Iterable[np.ndarray],
        metric_generators: typing.Dict[str, typing.Callable],
    ):
        # logits in shape (S x I x N) where S is noise samples, I iters, N examples
        for i, array in enumerate(input_source):
            last_array = array
            last_metrics = {}
            for fn_name, metric_fn in metric_generators.items():
                metric_name = f"{name}-{fn_name}"
                metric = self.job.cached(
                    lambda: metric_fn(array),
                    self.subdir,
                    f"{i}-{metric_name}.pt",
                    to_cpu=True,
                )
                last_metrics[metric_name] = metric
        # check curves of input_source at highest/lowest values of each metric, use random (last) rep
        self.plotter.plot_curves_by_rank(last_array, last_metrics)
        return list(last_metrics.keys())

    def list_metric_names(self):
        metric_names = set()
        dir = os.path.join(self.job.save_path, self.subdir)
        for f in os.listdir(dir):
            if f.endswith(".pt"):
                prefix = f.split("-")[0] + "-"
                metric_names.add(f[len(prefix) : -len(".pt")])
        return list(metric_names)

    def combine_replicates(self):
        # if collated file doesn't exist, load metrics and combine
        def combine(name):
            reps = []
            dir = os.path.join(self.job.save_path, self.subdir)
            for f in os.listdir(dir):
                if f.endswith(f"-{name}.pt"):
                    rep = torch.load(os.path.join(dir, f))
                    idx = int(f.split("-")[0])
                    reps.append(rep)
                    print(f, idx, name, transforms.stats_str(rep))
            return np.stack(reps, axis=0)

        metric_dict = {}
        for name in self.list_metric_names():
            metric = self.job.cached(
                lambda: combine(name), self.subdir, f"{name}.metric"
            )
            metric_dict[name] = metric
        return metric_dict

    def load_metrics(self):
        dir = os.path.join(self.job.save_path, self.subdir)
        files = os.listdir(dir)
        metric_dict = {}
        for f in files:
            if f.endswith(".metric"):
                metric = torch.load(
                    os.path.join(dir, f), map_location=torch.device("cpu")
                )
                metric_dict[f[: -len(".metric")]] = metric
                print(f, transforms.stats_str(metric))
        return metric_dict

    @staticmethod
    def filter_metrics(metric_dict, key):
        filtered_names = filter(lambda x: key in x, metric_dict.keys())
        return {n: metric_dict[n] for n in filtered_names}

    @staticmethod
    def apply(dict_metrics, transform_fn, suffix=""):
        if suffix != "":
            suffix = "-" + suffix
        output_metrics = {}
        for name, metric in dict_metrics.items():
            print(name, transforms.stats_str(metric))
            output_metrics[f"{name}{suffix}"] = transform_fn(metric)
        return output_metrics

    @staticmethod
    def average_over_samples(dict_metrics):
        return Metrics.apply(
            dict_metrics,
            lambda metric: np.median(metric, axis=-2)
            if len(metric.shape) == 3
            else metric,
        )

    @staticmethod
    def average_over_replicates(dict_metrics):
        return Metrics.apply(
            dict_metrics,
            lambda metric: np.median(metric, axis=0)
            if len(metric.shape) == 2
            else metric,
        )
