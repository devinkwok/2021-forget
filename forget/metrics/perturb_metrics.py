import typing
import numpy as np
from forget.perturb.perturb import Perturbation
from forget.metrics import transforms
from forget.metrics.metrics import Metrics


class PerturbMetrics(Metrics):
    def gen_metrics(self, perturbation: Perturbation):
        scale = self.extend_linear_scale(perturbation.scales)

        def logit_to_signed_prob():
            for logit in perturbation.logits():
                yield transforms.signed_prob(
                    transforms.softmax(logit), self.job.get_eval_labels()
                )

        s_prob_generators = {
            "first_forget": lambda x: transforms.last_always_true_index(x > 0, scale),
            "mean_prob": transforms.mean_prob,
        }
        self._gen_metrics(perturbation.name, logit_to_signed_prob(), s_prob_generators)

        def fraction_correct():
            for s_prob in logit_to_signed_prob():
                fraction_correct_samples = np.mean(s_prob > 0, axis=0)
                yield fraction_correct_samples

        fraction_generators = {
            "gibbs_error": lambda x: transforms.last_always_true_index(x > 0.5, scale),
        }
        self._gen_metrics(perturbation.name, fraction_correct(), fraction_generators)

    def extend_linear_scale(self, scale: np.ndarray):
        step_size = scale[-1] - scale[-2]
        new_item = [scale[-1] + step_size]
        scale = np.concatenate([scale, new_item])
        return scale
