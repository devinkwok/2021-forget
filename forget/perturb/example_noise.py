import abc
import math
import numpy as np
import torch
from forget.perturb.perturb import Perturbation


class ExamplePerturbation(Perturbation):
    def __init__(self, name, type, job):
        example_shape = job.example_shape
        assert len(example_shape) == 3  # assume last 3 dimensions are (C, H, W)
        if type == "channel":  # generate independent noise per channel
            self.example_shape = example_shape
            self.example_size = np.prod(example_shape)
        elif type == "greyscale":  # generate same noise for all channels
            self.example_shape = example_shape[1:]
            self.example_size = np.prod(example_shape[1:])
        else:
            raise ValueError(f"Example noise type {type} is not valid config")
        super().__init__(f"exnoise_{name}_{type}", job, True)

    def apply_perturbation(self, noise, scale_idx, scale, model, examples):
        perturbed_examples = self._apply_perturbation(noise, scale, examples)
        # only save if not already present
        self.job.cached(
            lambda: perturbed_examples[: self.job.batch_size].detach().cpu().numpy(),
            self.subdir,
            f"samplebatch_it{scale_idx}.pt",
        )
        return model, perturbed_examples

    @abc.abstractmethod
    def _apply_perturbation(self, noise, scale, examples):
        return examples


class ExampleIidGaussianNoise(ExamplePerturbation):
    def __init__(self, job):
        super().__init__("gauss", job.hparams["exnoise_gauss type"], job)

    def _apply_perturbation(self, noise, scale, examples):
        perturbed_examples = examples + noise * scale  # additive noise
        return perturbed_examples

    def gen_noise_sample(self):
        return torch.normal(0, 1.0, self.example_shape)


class ExampleIidEraseNoise(ExamplePerturbation):
    def __init__(self, job):
        super().__init__("erase", job.hparams["exnoise_erase type"], job)
        self.erased_value = 0

    def _apply_perturbation(self, noise, scale, examples):
        n_erase = int(math.ceil(self.example_size * scale))
        erase_mask = noise < n_erase
        perturbed_examples = examples.masked_fill(erase_mask, self.erased_value)
        return perturbed_examples

    def gen_noise_sample(self):
        return torch.randperm(self.example_size).reshape(self.example_shape)
