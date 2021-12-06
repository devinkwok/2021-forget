import time
import torch
import numpy as np
from forget.job import evaluate_one_batch


class NoisePerturbation:
    def __init__(self, job, filter_layer_names_containing=[]):
        self.job = job
        self.name_contains = filter_layer_names_containing
        if type(self.name_contains) is str:
            self.name_contains = [self.name_contains]
        self.noise_dist = self.job.hparams["noise distribution"]
        self.noise_type = self.job.hparams["noise type"]

    @property
    def subdir(self):
        return f'logits_noise_{self.noise_type}_{"-".join(self.name_contains)}'

    @property
    def scales(self):
        # always start from 0 to see if example was learned in trained model
        return np.linspace(
            0.0,
            float(self.job.hparams["noise scale max"]),
            int(self.job.hparams["noise num points"]),
        )

    def noise_samples(self):
        if self.noise_dist == "gaussian":
            noise_fn = self.sample_gaussians  # this allows other noise distributions
        else:
            raise ValueError(
                f"config value 'noise dist'={self.noise_dist} is undefined"
            )

        # save noise checkpoints
        for i in range(int(self.job.hparams["num noise samples"])):
            # only save if doesn't exist
            def gen_noise():  # name function so that job.cached() prints name
                return {
                    "type": self.noise_dist,
                    "replicate": i,
                    "model_state_dict": noise_fn(),
                }

            yield self.job.cached(gen_noise, "noise_" + self.noise_dist, f"noise{i}.pt")

    def noise_logits(self):
        # load dataset to CUDA
        examples, labels = zip(*self.job.get_eval_dataset())
        examples = torch.stack(examples, dim=0).cuda()
        labels = torch.tensor(labels).cuda()

        # load trained models and sample noise
        model_states = [
            ckpt["model_state_dict"]
            for ckpt, _ in self.job.load_checkpoints_by_epoch(-1)
        ]
        noise_states = [ckpt["model_state_dict"] for ckpt in self.noise_samples()]
        # cross product over samples x replicates
        for m, model in enumerate(model_states):
            for n, noise in enumerate(noise_states):
                # save logits over all noise scales
                def noise_logit():
                    logits, accuracies, scales = [], [], []
                    for noisy_model, scale in self.interpolate_noise(model, noise):
                        start_time = time.perf_counter()
                        output, accuracy = evaluate_one_batch(
                            noisy_model, examples, labels
                        )
                        accuracies.append(accuracy)
                        logits.append(output)
                        scales.append(scale)
                        print(
                            f"\ts={scale}, a={accuracy}, t={time.perf_counter() - start_time}"
                        )
                    return {
                        "type": self.noise_type,
                        "logit": torch.stack(logits, dim=0),
                        "scale": scales,
                        "accuracy": accuracies,
                    }

                self.job.cached(
                    noise_logit, self.subdir, f"logits-model{m}-noise{n}.pt"
                )

    def sample_gaussians(self):
        noise = self.job.get_model()
        noise.eval()
        with torch.no_grad():
            for param in noise.parameters():
                param.normal_(mean=0, std=1.0)
        return noise.state_dict()

    def apply_noise(self, model_state, noise_state, scale, combine_fn):
        with torch.no_grad():
            model = self.job.get_model(model_state)
            noise = self.job.get_model(noise_state)
            model.eval()
            noise.eval()
            for (name, param), param_noise in zip(
                model.named_parameters(), noise.parameters()
            ):
                if len(self.name_contains) == 0 or any(
                    x in name for x in self.name_contains
                ):
                    combine_fn(param, param_noise, scale)
            return model

    def interpolate_noise(self, model_state, noise_state):
        noise_type = self.job.hparams["noise type"]
        if noise_type == "additive":
            combine_fn = apply_additive_noise
        elif noise_type == "multiplicative":
            combine_fn = apply_multiplicative_noise
        else:
            raise ValueError(f"config value 'noise type'={noise_type} is undefined")
        # interpolate noise with model
        for scale in self.scales:
            noisy_model = self.apply_noise(model_state, noise_state, scale, combine_fn)
            yield noisy_model, scale


def apply_additive_noise(param, param_noise, scale):
    param.add_(param_noise * scale)


def apply_multiplicative_noise(param, param_noise, scale):
    param.mul_(1.0 + param_noise * scale)
