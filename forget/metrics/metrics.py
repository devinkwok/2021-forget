import os
import typing
from collections import OrderedDict
import numpy as np
import torch
from forget.job import Job
from forget.metrics import transforms


class Metrics:
    def __init__(self, job: Job, plotter):
        # filter batch by train/test examples
        self.job = job
        self.subdir = f"metrics-ep{self.job.n_epochs}"
        self.plotter = plotter
        self.it_split = (
            int(self.job.hparams["plot early late epoch split point"])
            * self.job.n_iter_per_epoch
        )

    def _gen_metrics(
        self,
        name: str,
        input_source: typing.Iterable[np.ndarray],
        metric_generators: typing.Dict[str, typing.Callable],
        do_plot_curves=True,
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
        if do_plot_curves:
            self.save_curves_by_rank(last_array, last_metrics)
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
        files.sort()
        metric_dict = OrderedDict()
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

    def save_curves_by_rank(self, scores, dict_metrics, n_rank=1):
        # for each metric, plot largest, middle, and smallest ranked example of first replicate/sample
        scores = scores.reshape(-1, scores.shape[-2], scores.shape[-1])[0]
        mid = scores.shape[-1] // 2 - n_rank // 2  # rank of middle example
        plot_name = ""
        selected, selected_ranks = {}, {}
        for name, metric in dict_metrics.items():
            # group dims to (RS... N) and take first element of RS
            metric = metric.reshape(-1, metric.shape[-1])[0]
            assert metric.shape[-1] == scores.shape[-1]
            ranks = np.argsort(metric)
            selected_ranks[name] = np.concatenate(
                [ranks[:n_rank], ranks[mid : mid + n_rank], ranks[-n_rank:]]
            )
            # swap axes to (lines, iters)
            selected[name] = scores[:, selected_ranks[name]].transpose(1, 0)
            plot_name = (
                name  # take any metric as a name to differentiate from other plots
            )
        group = (
            np.arange(3).repeat(n_rank).reshape(-1, 1).repeat(scores.shape[-2], axis=1)
        )
        group_names = ["Low", "Med", "High"]
        self.job.save_obj_to_subdir(
            {
                "lines": selected,
                "example_idx": selected_ranks,
                "metric_id": group,
                "metric_names": group_names,
            },
            self.subdir,
            f"{plot_name}-etc_curves.mdict",
        )
