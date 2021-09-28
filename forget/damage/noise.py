import os
import time
import torch
import numpy as np


def sample_noise(job):
    noise_dist = job.hparams['noise distribution']
    if noise_dist == 'gaussian':
        noise_fn = sample_gaussians  # this allows other noise distributions
    else:
        raise ValueError(f"config value 'noise dist'={noise_dist} is undefined")
    # save noise checkpoints
    noises = [noise_fn(job)
            for _ in range(int(job.hparams['num noise samples']))]
    for i, noise in enumerate(noises):
        job.save_obj_to_subdir(
            {
                'type': noise_dist,
                'replicate': i,
                'model_state_dict': noise
            },
            'noise_' + noise_dist, f'noise{i}.pt')

def eval_noise(job, name_contains=[]):
    if type(name_contains) is str:
        name_contains = [name_contains]
    noise_type = job.hparams['noise type']
    noise_ckpt_freq = int(job.hparams['noise checkpoint frequency'])

    # load dataset to CUDA
    examples, labels = zip(*job.get_eval_dataset())
    examples = torch.stack(examples, dim=0).cuda()
    labels = torch.tensor(labels).cuda()

    # load trained models and sample noise
    model_states = [ckpt['model_state_dict']
                    for ckpt, _ in job.load_checkpoints_by_epoch(-1)]
    noise_states = [ckpt['model_state_dict']
                    for ckpt, _ in job.load_checkpoints_from_dir(
                        'noise_' + job.hparams['noise distribution'])]
    # cross product over samples x replicates
    for m, model in enumerate(model_states):
        for n, noise in enumerate(noise_states):
            logits, accuracies, scales = [], [], []
            for i, (noisy_model, scale) in enumerate(
                    interpolate_noise(job, model, noise, name_contains)):
                start_time = time.perf_counter()
                output, accuracy = evaluate_one_batch(noisy_model, examples, labels)
                accuracies.append(accuracy)
                logits.append(output)
                scales.append(scale)
                print(f's={scale}, a={accuracy}, t={time.perf_counter() - start_time}')
                # save noise checkpoint
                if noise_ckpt_freq > 0 and ((i + 1) % noise_ckpt_freq == 0):
                    job.save_obj_to_subdir(
                        {
                            'noise_epoch': i,
                            'noise_scale': scale,
                            'model_state_dict': noisy_model.state_dict(),
                            'layer_name_contains': name_contains,
                        },
                        f'noise_{noise_type}_{"-".join(name_contains)}',
                        f'model={m}-noise={n}-epoch={i}.pt')
            # save logits over all noise scales
            job.save_obj_to_subdir(
                {
                    'type': noise_type,
                    'logit': torch.stack(logits, dim=0),
                    'scale': scales,
                    'accuracy': accuracies,
                },
                f'logits_noise_{noise_type}+{"-".join(name_contains)}',
                f'logits-model{m}-noise{n}.pt')

def sample_gaussians(job):
    noise = job.get_model()
    noise.eval()
    with torch.no_grad():
        for param in noise.parameters():
            param.normal_(mean=0, std=1.)
    return noise.state_dict()

def apply_noise(job, model_state, noise_state, scale, combine_fn, name_contains):
    with torch.no_grad():
        model = job.get_model(model_state)
        noise = job.get_model(noise_state)
        model.eval()
        noise.eval()
        for (name, param), param_noise in zip(
                    model.named_parameters(), noise.parameters()):
            if len(name_contains) == 0 or \
                    any(x in name for x in name_contains):
                combine_fn(param, param_noise, scale)
        return model

def apply_additive_noise(param, param_noise, scale):
    param.add_(param_noise * scale)

def apply_multiplicative_noise(param, param_noise, scale):
    param.mul_(1. + param_noise * scale)

def interpolate_noise(job, model_state, noise_state, name_contains):
    noise_type = job.hparams['noise type']
    if noise_type == 'additive':
        combine_fn = apply_additive_noise
    elif noise_type == 'multiplicative':
        combine_fn = apply_multiplicative_noise
    else:
        raise ValueError(f"config value 'noise type'={noise_type} is undefined")
    # linear scaling
    noise_scales = np.linspace(
        float(job.hparams["noise scale min"]),
        float(job.hparams["noise scale max"]),
        int(job.hparams["noise num points"]))
    # interpolate noise with model
    for scale in noise_scales:
        noisy_model = apply_noise(job, model_state, noise_state, scale,
                            combine_fn, name_contains)
        yield noisy_model, scale

def evaluate_one_batch(model, examples, labels):
    n_examples = labels.shape[0]
    with torch.no_grad():
        output = model(examples).detach()
        accuracy = torch.sum(torch.argmax(output, dim=1) == labels).float() / n_examples
    return output, accuracy.item()