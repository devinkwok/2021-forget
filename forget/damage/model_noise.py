import torch
from forget.damage.perturb import Perturbation


class ModelNoisePerturbation(Perturbation):
    def __init__(self, job, filter_layer_names_containing=[]):
        type = job.hparams["mnoise type"]
        if type == "add":
            self.combine_fn = self.apply_additive_noise
        elif type == "mult":
            self.combine_fn = self.apply_multiplicative_noise
        else:
            raise ValueError(f"Model noise type {type} is not valid config")

        self.name_contains = filter_layer_names_containing
        if isinstance(self.name_contains, str):
            self.name_contains = [self.name_contains]
        name = f'mnoise_{type}_{"-".join(self.name_contains)}'
        super().__init__(name, job, has_multiple_samples=True)

    def sample_gaussians(self):
        noise = self.job.get_model()
        noise.eval()
        with torch.no_grad():
            for param in noise.parameters():
                param.normal_(mean=0, std=1.0)
        return noise.state_dict()

    def gen_noise_sample(self):
        return self.sample_gaussians()

    def apply_perturbation(self, noise, scale, model, examples):
        noisy_model = super().interpolate_model_state(
            model.state_dict(), noise, scale, self.combine_fn, self.name_contains
        )
        return noisy_model, examples

    @staticmethod
    def apply_additive_noise(param, param_noise, scale):
        param.add_(param_noise * scale)

    @staticmethod
    def apply_multiplicative_noise(param, param_noise, scale):
        param.mul_(1.0 + param_noise * scale)
