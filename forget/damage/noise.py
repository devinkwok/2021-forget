import os
import time
import datetime
import torch
import numpy as np


def sample_and_eval_noisy_models(job):
    # load dataset to CUDA
    examples, labels = zip(*job.get_eval_dataset())
    examples = torch.stack(examples, dim=0).cuda()
    labels = torch.tensor(labels).cuda()

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
        job.save_obj_to_subdir(noise, 'noise_' + noise_type, f'noise{i}.pt')
    
    # save logits for sample/replicate
    for logits, scales, acc, m, n in eval_noisy_models(
            job, examples, labels, model_states, noises, combine_fn):
        print(f'Output m={m} n={n} t={datetime.datetime.now()}')
        job.save_obj_to_subdir(
            {'type': noise_type, 'logit': logits, 'scale': scales, 'accuracy': acc},
            'logits_noise_' + noise_type, f'logits-model{m}-noise{n}.pt')

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
    param.mul_(1. + param_noise * scale)

def eval_noisy_models(job, examples, labels, model_states, noises, combine_fn):
    n_examples = labels.shape[0]
    for m, model_state in enumerate(model_states):
        for n, noise in enumerate(noises):
            logits, accuracies = [], []
            # interpolate noisy models
            noise_scales = np.linspace(
                float(job.hparams["noise scale min"]),
                float(job.hparams["noise scale max"]),
                int(job.hparams["noise num points"]))
            for scale in noise_scales:
                start_time = time.perf_counter()
                noisy_model = apply_noise(job, model_state, noise, scale, combine_fn)
                # evaluate dataset
                with torch.no_grad():
                    output = noisy_model(examples).detach()
                    accuracy = torch.sum(torch.argmax(output, dim=1) == labels).float() / n_examples
                    accuracies.append(accuracy)
                    logits.append(output)
                print(f's={scale}, a={accuracy}, t={time.perf_counter() - start_time}')
            yield torch.stack(logits, dim=0), noise_scales, accuracies, m, n
