import typing
import numpy as np
from forget.postprocess.metrics import Metrics
from forget.postprocess import transforms


class PairReplicateMetrics(Metrics):
    def __init__(self, job, plotter, norm_order=1):
        super().__init__(job, plotter)
        self.norm_order = norm_order
        self.subdir = "pair_" + self.subdir

    def pair_train_logits(self):
        for i in range(self.job.n_replicates - 1):
            prob_diff = []
            for j in range(self.job.n_epochs + 1):
                logit_1 = self.job.load_from_replicate(
                    i, j, "eval_logits=", to_cpu=True
                )
                logit_2 = self.job.load_from_replicate(
                    i + 1, j, "eval_logits=", to_cpu=True
                )
                # norm over last dim C (class probabilities)
                vector_diff = transforms.softmax(logit_2) - transforms.softmax(logit_1)
                prob_diff.append(
                    np.linalg.norm(vector_diff, ord=self.norm_order, axis=-1)
                )
            # concatenate to (1 x EI+1 x N) where E is epochs and I is iters
            prob_diff = np.expand_dims(np.concatenate(prob_diff, axis=0), axis=0)
            assert prob_diff.shape == (
                1,
                self.job.n_epochs * self.job.n_iter_per_epoch
                + 1,  # add 1 for epoch 0 (untrained) logit
                self.job.n_eval_examples,
            ), prob_diff.shape
            yield prob_diff

    def gen_metrics_from_training(self):
        # wrappers for transformations to fill in some args
        def mean_pdiff(pdiff):
            return transforms.mean_prob(pdiff)

        def early_mean_pdiff(pdiff):
            return transforms.mean_prob(pdiff[:, : self.it_split, :])

        def late_mean_pdiff(pdiff):
            return transforms.mean_prob(pdiff[:, self.it_split :, :])

        def pdiff_peak_iter(pdiff):
            return np.argmax(pdiff, axis=-2)  # max iter over EI+1 dim

        def pdiff_centerOfMass(pdiff):
            return transforms.center_of_mass(pdiff)

        metric_generators = {
            "early_mean_diff": early_mean_pdiff,
            "late_mean_diff": late_mean_pdiff,
            "peak_iter": pdiff_peak_iter,
            "centerOfMass": pdiff_centerOfMass,
        }
        return self.gen_metrics("pair", self.pair_train_logits(), metric_generators)
