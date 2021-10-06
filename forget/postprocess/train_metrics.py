import os
import time
import typing
import torch
import numpy as np
from forget.postprocess.plot_metrics import PlotMetrics


NO_CLASS_IDX = -1


def top_k(
        output_prob: np.ndarray, tgt_labels: np.ndarray, k=5
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Turns model outputs into a list of the top k scoring classes.

    Args:
        output_prob (np.ndarray): concatenated tensor of output probabilities (after Softmax),
            dimensions $(N \times C)$ where $N$ is number of examples, $C$ is number of classes.
        tgt_labels (np.ndarray): true classification labels given as
            integers between $0$ and $C - 1$ in a tensor of dimension $(N)$.
        k (int, optional): maximum number of classes to include. Defaults to 5.

    Returns:
        Tuple[np.ndarray, np.ndarray]: classification class indexes and probabilities
            for samples in output_prob, ordered by largest probability.
            Dimension of each array is $N \times k$.
            Probabilities are multiplied by -1 if class differs from tgt_labels.
    """
    n = output_prob.shape[0]
    top_k_classes = np.flip(np.argsort(output_prob, axis=1), axis=1)[:, :k]
    idx = np.arange(n).reshape(n, 1).repeat(k, axis=1)
    scores = output_prob[idx, top_k_classes]
    tgt_labels = tgt_labels.reshape(n, 1).repeat(k, axis=1)
    scores[top_k_classes != tgt_labels] *= -1
    return top_k_classes, scores


def n_cumulative_top_classes(top_classes_by_iter: typing.List[np.ndarray]) -> np.ndarray:
    """Counts the cumulative number of unique top classes for outputs of
    outputs_to_top_classes over multiple iterations.

    Args:
        top_classes (np.ndarray): list of outputs from outputs_to_top_classes
            over multiple iterations, where each iteration is
            called on same examples in same batch order.
            List should be ordered from earliest to latest iteration.

    Returns:
        np.ndarray: the number of unique classes assigned to each sample as iterations increase.
            Dimensions are $(N \times T)$ where $T$ is number of iterations.
    """
    n_cumulative = []
    cumulative = np.full_like(top_classes_by_iter[0], NO_CLASS_IDX)
    for top_classes in top_classes_by_iter:
        cumulative = np.concatenate([cumulative, top_classes], axis=1)
        n_unique = [len(np.unique(sample)) for sample in cumulative]
        n_cumulative.append(np.array(n_unique) - 1)  # don't count NO_CLASS_IDX
    return np.stack(n_cumulative, axis=1)


def signed_prob(output_prob: np.ndarray, tgt_labels: np.ndarray) -> np.ndarray:
    """
    Args:
        output_prob (np.ndarray): output probabilities (after Softmax) with
            dimensions $(\dots, N \times C)$ where $N$ is number of examples,
            $C$ is number of classes.
        tgt_labels (np.ndarray): true classification labels given as
            integers between $0$ and $C - 1$ in a tensor of dimension $(N)$.

    Returns:
        (np.ndarray): probability of the correct class from `tgt_labels`,
            multiplied by -1 if the correct class was not the argmax,
            dimension $(\dots, N)$.
    """
    probabilities = output_prob[..., np.arange(len(tgt_labels)), tgt_labels]
    class_labels = np.argmax(output_prob, axis=-1)
    correct_mask = (class_labels == tgt_labels) * 2 - 1
    signed_prob = probabilities * correct_mask
    return signed_prob


def mean_prob(output_prob_by_iter: np.ndarray, divide_by_iters=True) -> np.ndarray:
    """
    Args:
        output_prob_by_iter (np.ndarray): signed_prob scores
            with dimensions $(\dots, I \times N)$.
            where each iteration $I$ is over same examples in same batch order.
        divide_by_iters (bool, optional): If True, divide area under curve
            by number of iterations to normalize between 0 and 1. Defaults to True.

    Returns:
        np.ndarray: mean of $N$ over $I$, dimensions $(\dots, N)$.
    """
    auc = np.sum(np.abs(output_prob_by_iter), axis=-2)
    if divide_by_iters:
        auc = auc / output_prob_by_iter.shape[-2]
    return auc


def diff_norm(output_prob_by_iter: np.ndarray,
            norm_power=1, divide_by_iters=True) -> np.ndarray:
    """Takes difference of output probabilities between successive iterations,
    and calculates a norm on this differencing operation.

    Args:
        output_prob_by_iter (np.ndarray): signed_prob scores
            with dimensions $(\dots, I \times N)$,
            where each iteration $I$ is over same examples in same batch order.
        norm_power (int, optional): Type of norm (e.g. 1 is abs diff, 2 is mean squared difference).
            If less than 1, returns infinity norm (max difference). Defaults to 1.
        divide_by_iters (bool, optional): If True, divide result by $N$.

    Returns:
        np.ndarray: array of $(\dots, N)$ mean differences.
    """
    output_prob_by_iter = np.abs(output_prob_by_iter)
    diff = np.abs(output_prob_by_iter[..., :-1, :] - output_prob_by_iter[..., 1:, :])
    if norm_power <= 0:
        norm = np.max(diff, axis=-2)
    else:
        norm = (np.sum(diff**norm_power, axis=-2) / diff.shape[-2])**(1 / norm_power)
    if divide_by_iters:
        norm = norm / output_prob_by_iter.shape[-2]
    return norm

def identity(array):
    """For stacking using transform()
    """
    return array

def softmax(logits):
    with torch.no_grad():
        return torch.softmax(logits, dim=-1).detach().numpy()

def forgetting_events(output_prob_by_iter, iter_mask=None) -> np.ndarray:
    pass #TODO

def mask_iter_by_batch(train_batch_size, example_start_idx, example_end_idx):
    pass #TODO

def stats_str(array):
    return '<{:0.4f}|{:0.4f}|{:0.4f}> {}'.format(
        np.min(array), np.mean(array), np.max(array), array.shape)

class GenerateMetrics():

    def __init__(self, job, force_generate=False):
        # filter batch by train/test examples
        self.job = job
        self.force_generate = force_generate
        self.iter_mask = mask_iter_by_batch(self.job.hparams['batch size'],
                        0, int(self.job.hparams['eval number of train examples']))
        self.labels = np.array([y for _, y in self.job.get_eval_dataset()])

    def _transform(self, name, source, transform_fn):
        start_time = time.perf_counter()
        output = transform_fn(source)
        print(f'\t{stats_str(output)} t={time.perf_counter() - start_time} {name}')
        return output

    def transform_inplace(self, source_file_iterable, transform_fn):
        for source_file, source in source_file_iterable:
            source_path, ext = os.path.splitext(source_file)
            out_file = source_path + '-' + transform_fn.__name__ + ext
            if not os.path.exists(out_file) or self.force_generate:
                print(f'Generating in-place {transform_fn.__name__} ' + \
                        'from {source_file_iterable.__name__}:')
                output = self._transform(source_file, source, transform_fn)
                torch.save(output, out_file)
            yield torch.load(out_file, map_location=torch.device('cpu'))

    def transform_collate(self, source_iterable, transform_fn):
        name = f'{source_iterable.__name__}-{transform_fn.__name__}.metric'
        out_file = os.path.join(self.job.save_path, 'metrics', name)
        # check if output of transform_fn already exists for source_iterable
        if not os.path.exists(out_file) or self.force_generate:
            print(f'Generating collated {transform_fn.__name__} ' + \
                    'from {source_iterable.__name__}:')
            outputs = [self._transform(i, source, transform_fn)
                        for i, source in enumerate(source_iterable)]
            torch.save(np.stack(outputs, axis=0), out_file)
        return torch.load(out_file, map_location=torch.device('cpu'))

    def replicate_by_epoch(self, model_dir, prefix='eval_logits'):  # source_iter
        # logits are saved after epochs, hence index from 1
        for i in range(1, self.job.n_epochs + 1):
            file = os.path.join(model_dir, f'{prefix}={i}.pt')
            outputs = torch.load(file, map_location=torch.device('cpu'))
            yield file, outputs

    def noise_by_replicate():  # source_iter
        pass #TODO

    def batch_forgetting(self, output_prob_by_iter):  # as implemented by Toneva, same as Nikhil'ss
        return forgetting_events(output_prob_by_iter, batch_mask=self.batch_mask)

    def _train_eval_filter(self, ndarray, split_type):  # applies to metrics with N as last dim
        split_point = int(self.job.hparams['eval number of train examples'])
        if self.include_examples == 'train':
            return ndarray[..., :split_point]
        elif self.include_examples == 'test':
            return ndarray[..., split_point:]
        else:
            return ndarray

    def signed_prob(self, x):
        return signed_prob(x, self.labels)

    def gen_train_metrics_by_epoch(self):
        plotter = PlotMetrics(self.job)
        # R * E * (I x N x C) (list of R iterators over E)
        # R is replicates, E is epochs, I iters, N examples, C classes
        softmaxes = [self.transform_inplace(self.replicate_by_epoch(r),
                        softmax) for r, _ in self.job.replicate_dirs()]
        # collate over E and transform over C to get R * (E x I x N)
        s_prob = [self.transform_collate(s, self.signed_prob) for s in softmaxes]
        s_prob = np.stack(s_prob, axis=0)  # stack to (R x E x I x N)
        n_epoch, n_iter, n_example = s_prob.shape[-3], s_prob.shape[-2], s_prob.shape[-1]
        # plot curves for selected examples
        curves = s_prob[..., np.arange(0, 10000, 1000)]
        # reshape to (R x EI x N)
        curves = curves.reshape(-1, n_epoch * n_iter, n_example)
        # transpose to (N x R x EI)
        curves = curves.transpose(axes=(2, 0, 1))
        plotter.plot_score_curves(curves)

        metrics_by_epoch = {  # each metric is R x E x N
            'sgd_mean_prob': self.transform_inplace(s_prob, mean_prob),
            'sgd_diff_norm': self.transform_inplace(s_prob, diff_norm),
            # 'sgd_forgetting': self.generate_metrics(s_prob, forgetting_events),
            # 'sgd_batch_forgetting': self.generate_metrics(s_prob, self.batch_forgetting),
        }
        # filter by train/test/all
        for include in ['all', 'train', 'test']:
            name = f'-{include}'
            # label distribution
            plotter.plot_class_counts(name,
                    self._train_eval_filter(self.labels, include))
            # plot mean over epochs (R x N)
            plotter.plot_metrics(metrics_by_epoch, name,
                filter=lambda x: np.mean(self._train_eval_filter(x, include), axis=1))
            # plot by epoch
            for i in range(n_epoch):
                plotter.plot_metrics(metrics_by_epoch, f'{name}-ep{i}',
                    filter=lambda x: self._train_eval_filter(x[:, i, :], include))
