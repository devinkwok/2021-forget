import typing
import numpy as np
from forget.damage.perturb import Perturbation
from forget.postprocess import transforms
from forget.postprocess.metrics import Metrics


class SingleReplicateMetrics(Metrics):
    def __init__(self, job, plotter):
        super().__init__(job, plotter)
        # TODO fix mask to account for randomized example order
        # self.iter_mask = mask_iter_by_batch(
        #     int(self.job.hparams["batch size"]),
        #     len(self.job.get_train_dataset()),
        #     0,
        #     len(self.labels),
        # )
        # self.iter_mask = np.concatenate(
        #     [self.iter_mask[1:], self.iter_mask[:1]], axis=0
        # )

    def gen_metrics_from_training(self):
        # wrappers for transformations to fill in some args
        def early_mean_prob(s_prob):
            return transforms.mean_prob(s_prob[:, : self.it_split, :])

        def late_mean_prob(s_prob):
            return transforms.mean_prob(s_prob[:, self.it_split :, :])

        def batch_forgetting(s_prob):
            return transforms.forgetting_events(s_prob, iter_mask=self.iter_mask)

        def centerOfMass(s_prob):
            return transforms.center_of_mass(np.abs(s_prob))

        def mean_forget_iter(s_prob):
            return transforms.center_of_mass(
                transforms.forgetting_events(s_prob, return_count=False)
            )

        metric_generators = {
            "early_mean_prob": early_mean_prob,
            "late_mean_prob": late_mean_prob,
            "diff_norm": transforms.diff_norm,  # TODO diff_norm uses too much memory?
            "forget": transforms.forgetting_events,
            # "batch_forget": batch_forgetting,  # TODO account for randomized example order
            "first_learn": transforms.first_learn,
            "centerOfMass": centerOfMass,
            "mean_forget_iter": mean_forget_iter,
        }
        return self.gen_metrics("train", self.train_logits(), metric_generators)

    def train_logits(self):
        for i, _ in enumerate(self.job.replicates()):
            s_prob = []
            for j in range(self.job.n_epochs + 1):
                logit = self.job.load_from_replicate(i, j, "eval_logits=", to_cpu=True)
                s_prob += [
                    transforms.signed_prob(
                        transforms.softmax(logit), self.job.get_eval_labels()
                    )
                ]
            # concatenate to (1 x EI+1 x N) where E is epochs and I is iters
            s_prob = np.expand_dims(np.concatenate(s_prob, axis=0), axis=0)
            assert s_prob.shape == (
                1,
                self.job.n_epochs * self.job.n_iter_per_epoch
                + 1,  # add 1 for epoch 0 (untrained) logit
                self.job.n_eval_examples,
            ), s_prob.shape
            yield s_prob

    def gen_metrics_from_perturbation(self, perturbation: Perturbation):
        def logit_to_signed_prob():
            for logit in perturbation.logits():
                yield transforms.signed_prob(
                    transforms.softmax(logit), self.job.get_eval_labels()
                )

        scale = self.extend_linear_scale(perturbation.scales)
        metric_generators = {
            "first_forget": lambda x: transforms.first_forget(x, scale),  # wrapper
            "mean_prob": transforms.mean_prob,
        }
        return self.gen_metrics(
            perturbation.name, logit_to_signed_prob(), metric_generators
        )

    def extend_linear_scale(self, scale: np.ndarray):
        step_size = scale[-1] - scale[-2]
        new_item = [scale[-1] + step_size]
        scale = np.concatenate([scale, new_item])
        return scale
