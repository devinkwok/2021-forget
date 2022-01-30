import numpy as np
import torch
from forget.damage.perturb import Perturbation


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


class ExampleIidGaussianNoise(ExamplePerturbation):
    def __init__(self, job):
        super().__init__("gauss", job.hparams["exnoise_gauss type"], job)

    def apply_perturbation(self, noise, scale, model, examples):
        perturbed_examples = examples + noise * scale  # additive noise
        return model, perturbed_examples

    def gen_noise_sample(self):
        return torch.normal(0, 1.0, self.example_shape)


class ExampleIidEraseNoise(ExamplePerturbation):
    def __init__(self, job):
        super().__init__("erase", job.hparams["exnoise_erase type"], job)
        self.erased_value = 0

    def apply_perturbation(self, noise, scale, model, examples):
        n_erase = int(math.ceil(self.example_size * scale))
        erase_mask = noise < n_erase
        perturbed_examples = examples.masked_fill(erase_mask, self.erased_value)
        return model, perturbed_examples

    def gen_noise_sample(self):
        return torch.randperm(self.example_size).reshape(self.example_shape)
