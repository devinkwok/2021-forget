import os
import torch
import numpy as np
from itertools import product
from open_lth.models import registry
from open_lth.foundations import hparams
from pathlib import Path


def sample_and_eval_noisy_models(job):
    # load dataset to CUDA
    examples = torch.stack([x for x, _ in job.get_eval_dataset()], dim=0)
    examples = examples.cuda()

    noise_type = job.hparams['noise type']
    if noise_type == 'additive':
        sample_noise = sample_gaussians
        combine_fn = apply_additive_noise
    elif noise_type == 'multiplicative':
        sample_noise = sample_gaussians
        combine_fn = apply_multiplicative_noise
    else:
        raise ValueError(f"config value 'noise type'={noise_type} is undefined")

    # load trained models and sample noise
    model_states = load_model_states(job)
    noises = [sample_noise(job)
            for _ in range(int(job.hparams['num noise samples']))]

    # save noise checkpoints
    for i, noise in enumerate(noises):
        job.save_obj_to_subdir(noise, 'noise_' + noise_type, f'noise{i}')
    
    # save logits for sample/replicate
    for logits, m, n in eval_noisy_models(
            job, examples, model_states, noises, combine_fn):
        job.save_obj_to_subdir(
            logits, 'logits_noise_' + noise_type, f'logits-model{m}-noise{n}')

def load_model_states(job):
    model_states = []
    ckpt_id = job.hparams['num epochs']
    for dir in job.replicate_dirs():
        model = torch.load(os.path.join(dir, f'epoch={ckpt_id}.pt'))
        model_states.append(model['model_state_dict'])
    return model_states

def sample_gaussians(job):
    noise = job.get_model()
    noise.eval()
    with torch.no_grad():
        for param in noise.parameters():
            param.normal_(mean=0, std=1.)
    return noise

def apply_noise(job, model_state, noise, scale, combine_fn):
    with torch.no_grad():
        model = job.get_model()
        model.load_state_dict(model_state)
        model.eval()
        for param, param_noise in zip(model.parameters(), noise.parameters()):
            combine_fn(param, param_noise, scale)
        return model

def apply_additive_noise(param, param_noise, scale):
    param.add_(param_noise * scale)

def apply_multiplicative_noise(param, param_noise, scale):
    param.multiply_(1. + param_noise * scale)

def eval_noisy_models(job, examples, model_states, noises, combine_fn):
    for m, model_state in enumerate(model_states):
        for n, noise in enumerate(noises):
            # interpolate noisy models
            noise_scales = np.linspace(
                float(job.hparams["noise scale min"]),
                float(job.hparams["noise scale max"]),
                int(job.hparams["noise num points"]))
            for scale in noise_scales:
                noisy_model = apply_noise(job, model_state, noise, scale, combine_fn)
                # evaluate dataset
                with torch.no_grad():
                    yield noisy_model(examples).detach(), m, n
