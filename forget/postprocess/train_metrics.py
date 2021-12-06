import os
import time
import torch
import numpy as np
from forget.postprocess.transforms import *
from forget.postprocess.plot_metrics import PlotMetrics


class GenerateMetrics:
    def __init__(self, job, force_generate=False):
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

    # wrappers for transformations to fill in some args
    def batch_forgetting(
        self, output_prob_by_iter
    ):  # as implemented by Toneva, same as Nikhil'ss
        return forgetting_events(output_prob_by_iter, iter_mask=self.iter_mask)

    # wrapper
    def sgn_prob(self, x):
        return signed_prob(softmax(x), self.labels)

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
        return metrics, n_epoch, n_iter

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

    def gen_noise_metrics(self, noise_dir, noise_scale):
        # labels for noise scale, need I+1 items
        last_scale_item = [noise_scale[-1] + noise_scale[1]]
        noise_scale = np.concatenate([noise_scale, last_scale_item])
        # wrapper
        def noise_first_forget(x):
            return first_forget(x, noise_scale)
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
                s_prob, noise_first_forget, prefix="noise"
            ),
            "noise_mean_prob": self.transform(s_prob, mean_prob, prefix="noise"),
        }
        plotter = PlotMetrics(self.job)  # plots for manual validation
        plotter.plot_curves_by_rank(
            s_prob, metrics
        )  # check sgn_prob curves at highest/lowest metric
        return metrics

    # source_iterable over pruning logits by replicate
    def pruned_logits(self, prune_dir):  # source_iter
        for i in range(0, self.job.n_replicates):
            file = f"logits-model{i}.pt"

            def load_pruned():
                return torch.load(
                    os.path.join(self.job.save_path, prune_dir, file),
                    map_location=torch.device("cpu"),
                )['logit']

            yield load_pruned

    def gen_prune_metrics(self, prune_dir, prune_scale):
        # labels for noise scale, need I+1 items
        last_scale_item = [prune_scale[-1] + prune_scale[1]]
        prune_scale = np.concatenate([prune_scale, last_scale_item])
        # wrapper
        def prune_first_forget(x):
            return first_forget(x, prune_scale)
        # R is replicates, I iters, N examples
        print("Loading signed probabilities...")
        # collate over R to get (R x I x N)
        s_prob = self.transform_collate(self.pruned_logits(prune_dir),
                self.sgn_prob,
                prefix=prune_dir,
                subdir=prune_dir
            )
        print("Loading metrics...")
        # summarizes over I for (R x N)
        metrics = {
            "prune_first_forget": self.transform(
                s_prob, prune_first_forget, prefix="prune"
            ),
            "prune_mean_prob": self.transform(s_prob, mean_prob, prefix="prune"),
        }
        plotter = PlotMetrics(self.job)  # plots for manual validation
        plotter.plot_curves_by_rank(
            s_prob, metrics
        )  # check sgn_prob curves at highest/lowest metric
        return metrics
