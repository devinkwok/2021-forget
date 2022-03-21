import numpy as np
from forget.metrics import transforms
from forget.metrics.metrics import Metrics


class TrainMetrics(Metrics):
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

    def gen_metrics(self):
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
            "first_learn": lambda x: transforms.first_always_true_index(x > 0),
            "centerOfMass": centerOfMass,
            "mean_forget_iter": mean_forget_iter,
        }
        self._gen_metrics("train", self.train_logits(), metric_generators)

    def train_logits(self):
        for i, _ in enumerate(self.job.replicates()):
            s_prob = []
            for j in range(self.job.n_epochs + 1):
                logit = self.job.load_from_replicate(i, j, "logits-ep", to_cpu=True)
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
