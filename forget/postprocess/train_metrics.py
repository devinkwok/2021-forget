import os
import time
import torch
import numpy as np
from forget.postprocess.transforms import *
from forget.postprocess.plot_metrics import PlotMetrics


class GenerateMetrics:
    def __init__(self, job, noise_scale, force_generate=False):
        # filter batch by train/test examples
        self.job = job
        self.force_generate = force_generate
        eval_dataset = self.job.get_eval_dataset()
        self.labels = np.array([y for _, y in eval_dataset])
        self.iter_mask = mask_iter_by_batch(
            int(self.job.hparams["batch size"]),
            len(self.job.get_train_dataset()),
            0,
            len(self.labels),
        )
        # shift iter_mask back by 1 iter, this is because logits were recorded after SGD, not before
        # and Toneva counts forgetting based on example accuracy before SGD update
        self.iter_mask = np.concatenate(
            [self.iter_mask[1:], self.iter_mask[:1]], axis=0
        )
        self.split_point = int(self.job.hparams["eval number of train examples"])
        # labels for noise scale, need I+1 items
        last_scale_item = [noise_scale[-1] + noise_scale[1]]
        self.scale = np.concatenate([noise_scale, last_scale_item])
        self.early_late_split = int(
            self.job.hparams["plot early late epoch split point"]
        )

    def _transform(self, name, source, transform_fn):
        start_time = time.perf_counter()
        output = transform_fn(source)
        print(f"\t{stats_str(output)} t={time.perf_counter() - start_time} {name}")
        return output

    def transform(self, source, transform_fn, prefix="", subdir="metrics"):
        def generate():
            return self._transform(prefix, source, transform_fn)

        filename = f"{prefix}-{transform_fn.__name__}.metric"
        return self.job.cached(
            generate, subdir, filename, overwrite=self.force_generate, to_cpu=True
        )

    # source_iterable contains functions which, when called, gives object for transform
    def transform_collate(
        self, source_iterable, transform_fn, prefix="", subdir="metrics"
    ):
        def collate():
            outputs = [
                self._transform(str(i), source_fn(), transform_fn)
                for i, source_fn in enumerate(source_iterable)
            ]
            return np.stack(outputs, axis=0)

        filename = f"{prefix}-{transform_fn.__name__}.metric"
        return self.job.cached(
            collate, subdir, filename, overwrite=self.force_generate, to_cpu=True
        )

    # source_iterable over logits by epoch
    def replicate_by_epoch(self, model_dir, prefix="eval_logits"):  # source_iter
        # logits are saved after epochs, hence index from 1
        for i in range(1, self.job.n_epochs + 1):
            file = f"{prefix}={i}.pt"

            def load_epoch():
                return torch.load(
                    os.path.join(self.job.save_path, model_dir, file),
                    map_location=torch.device("cpu"),
                )

            yield load_epoch

    # source_iterable over noise logits by noise sample
    def noise_by_sample(self, noise_dir, replicate_name):  # source_iter
        for i in range(int(self.job.hparams["num noise samples"])):
            file = f"logits-{replicate_name}-noise{i}.pt"

            def load_noise_sample():
                outputs = torch.load(
                    os.path.join(self.job.save_path, noise_dir, file),
                    map_location=torch.device("cpu"),
                )
                return outputs["logit"]

            yield load_noise_sample

    # wrappers for transformations to fill in some args
    def batch_forgetting(
        self, output_prob_by_iter
    ):  # as implemented by Toneva, same as Nikhil'ss
        return forgetting_events(output_prob_by_iter, iter_mask=self.iter_mask)

    # wrapper
    def sgn_prob(self, x):
        return signed_prob(softmax(x), self.labels)

    # wrapper
    def first_forget(self, x):
        return first_forget(x, self.scale)

    def gen_train_metrics(self):
        # R is replicates, E is epochs, I iters, N examples, C classes
        print("Loading signed probabilities...")
        # iterate over R, collate over E to get R * (E x I x N)
        s_prob = [
            self.transform_collate(
                self.replicate_by_epoch(subdir),
                self.sgn_prob,
                prefix=subdir,
                subdir=subdir,
            )
            for _, subdir in self.job.replicates()
        ]
        s_prob = np.stack(s_prob, axis=0)  # stack to (R x E x I x N)
        n_epoch, n_iter, n_example = (
            s_prob.shape[-3],
            s_prob.shape[-2],
            s_prob.shape[-1],
        )
        # merge to (R x EI x N)
        s_prob = s_prob.reshape(-1, n_epoch * n_iter, n_example)
        # compute metrics over I to get (R x N)
        print("Loading metrics...")
        it_split = self.early_late_split * n_iter
        metrics = {
            "early_mean_prob": self.transform(
                s_prob[:, :it_split, :], mean_prob, prefix="early"
            ),
            "late_mean_prob": self.transform(
                s_prob[:, it_split:, :], mean_prob, prefix="late"
            ),
            # TODO diff_norm uses too much memory
            # 'diff_norm': self.transform(s_prob, diff_norm, prefix='train'),
            "forget": self.transform(s_prob, forgetting_events, prefix="train"),
            "batch_forget": self.transform(
                s_prob, self.batch_forgetting, prefix="train"
            ),
            "first_learn": self.transform(s_prob, first_learn, prefix="train"),
        }
        plotter = PlotMetrics(self.job)  # plots for manual validation
        plotter.plot_curves_by_rank(
            s_prob, metrics
        )  # check sgn_prob curves at highest/lowest metric
        return metrics

    def gen_noise_metrics(self, noise_dir):
        # R is replicates, S is noise samples, I iters, N examples, C classes
        print("Loading signed probabilities...")
        # iterate over R, collate over S to get R * (S x I x N)
        s_prob = [
            self.transform_collate(
                self.noise_by_sample(noise_dir, model_dir),
                self.sgn_prob,
                prefix=model_dir,
                subdir=noise_dir,
            )
            for _, model_dir in self.job.replicates()
        ]
        s_prob = np.stack(s_prob, axis=0)  # stack to (R x S x I x N)
        print("Loading metrics...")
        # summarizes over I for (R x S x N)
        metrics = {
            "noise_first_forget": self.transform(
                s_prob, self.first_forget, prefix="noise"
            ),
            "noise_mean_prob": self.transform(s_prob, mean_prob, prefix="noise"),
        }
        plotter = PlotMetrics(self.job)  # plots for manual validation
        plotter.plot_curves_by_rank(
            s_prob, metrics
        )  # check sgn_prob curves at highest/lowest metric
        return metrics
