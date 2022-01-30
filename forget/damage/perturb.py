import abc
import time
import numpy as np
import torch
from forget.job import evaluate_one_batch
from forget.postprocess.transforms import stats_str


class Perturbation:
    def __init__(self, name: str, job, has_multiple_samples):
        self.name = name
        self.job = job
        self.n_samples = 1
        if has_multiple_samples:
            self.n_samples = int(self.job.hparams[f"{self.name} n samples"])

    @property
    def subdir(self):
        return f"logits_{self.name}-ep{self.job.n_epochs}"

    @property
    def scales(self):
        scale_min = 0.0
        if f"{self.name} scale min" in self.job.hparams:
            scale_min = float(self.job.hparams[f"{self.name} scale min"])
        return np.linspace(
            scale_min,
            float(self.job.hparams[f"{self.name} scale max"]),
            int(self.job.hparams[f"{self.name} n points"]),
        )

    @abc.abstractmethod
    def apply_perturbation(self, noise, scale, model, examples):
        return model, examples

    @abc.abstractmethod
    def gen_noise_sample(self):
        return None

    def gen_logits(self):
        for _ in self.logits():
            pass

    def logits(self):
        # load dataset to CUDA
        examples, labels = zip(*self.job.get_eval_dataset())
        examples = torch.stack(examples, dim=0).cuda()
        labels = torch.tensor(labels).cuda()

        # load trained models and sample noise
        for replicate, ckpt in enumerate(
            self.job.load_checkpoints_by_epoch(self.job.n_epochs)
        ):
            model = self.job.get_model(state_dict=ckpt["model_state_dict"])

            def perturb_logit():
                logits, accuracies = [], []
                for i in range(self.n_samples):
                    per_noise_logits = []
                    for scale in self.scales:
                        start_time = time.perf_counter()
                        noise = self.gen_noise_sample()
                        if type(noise) == torch.Tensor:  # send to CUDA if tensor
                            noise = noise.cuda()
                        perturb_model, perturb_examples = self.apply_perturbation(
                            noise, scale, model, examples
                        )
                        output, accuracy = evaluate_one_batch(
                            perturb_model, perturb_examples, labels
                        )
                        accuracies.append(accuracy)
                        per_noise_logits.append(output)
                        print(
                            f"\t{i} s={scale}, a={accuracy}, t={time.perf_counter() - start_time}"
                        )
                    logits.append(torch.stack(per_noise_logits, dim=0))
                logits = torch.stack(logits, dim=0).detach().cpu()
                assert logits.shape == (
                    self.n_samples,
                    len(self.scales),
                    len(examples),
                    self.job.n_unique_labels,
                ), logits.shape
                return logits

            yield self.job.cached(
                perturb_logit, self.subdir, f"logits-m{replicate}.pt", to_cpu=True
            )

    def interpolate_model_state(
        self, source_state, target_state, scale, combine_fn, layer_name_contains
    ):
        with torch.no_grad():
            model = self.job.get_model(source_state)
            noise = self.job.get_model(target_state)
            model.eval()
            noise.eval()
            for (name, param), param_noise in zip(
                model.named_parameters(), noise.parameters()
            ):
                if len(layer_name_contains) == 0 or any(
                    x in name for x in layer_name_contains
                ):
                    combine_fn(param, param_noise, scale)
            return model
