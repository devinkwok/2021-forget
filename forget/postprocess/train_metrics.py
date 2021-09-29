import os
import time
import typing
import torch
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


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

def outputs_to_correctness(output_prob: np.ndarray, tgt_labels: np.ndarray) -> np.ndarray:
    """
    Args:
        output_prob (np.ndarray): concatenated tensor of outputs (after Softmax)
            dimensions $(N \times C)$ where $N$ is number of examples, $C$ is number of classes.
        tgt_labels (np.ndarray): true classification labels given as
            integers between $0$ and $C - 1$ in a tensor of dimension $(N)$.

    Returns:
        (np.ndarray): probability of the correct class from `tgt_labels`,
            multiplied by -1 if the correct class was not the argmax,
            dimension $(N)$.
    """
    probabilities = output_prob[np.arange(len(tgt_labels)), tgt_labels]
    class_labels = np.argmax(output_prob, axis=1)
    probabilities[class_labels != tgt_labels] *= -1
    return probabilities

def top_1_auc(output_prob_by_iter: typing.List[np.ndarray], divide_by_iters=True) -> np.ndarray:
    """
    Args:
        output_prob_by_iter (typing.List[np.ndarray]): list of outputs from
            outputs_to_correctness over multiple iterations.
        divide_by_iters (bool, optional): If True, divide area under curve
            by number of iterations to normalize between 0 and 1. Defaults to True.

    Returns:
        np.ndarray: array of $N$ area under curve floats.
    """
    auc = np.sum(np.abs(output_prob_by_iter), axis=0)
    if divide_by_iters:
        auc = auc / output_prob_by_iter.shape[0]
    return auc


def diff_norm(output_prob_by_iter: typing.List[np.ndarray],
            norm_power=1, divide_by_iters=True) -> np.ndarray:
    """Takes difference of output probabilities between successive iterations,
    and calculates a norm on this differencing operation.

    Args:
        output_prob_by_iter (typing.List[np.ndarray]): list of outputs from
            outputs_to_correctness over multiple iterations,
            where each iteration is called on same examples in same batch order.
            List should be ordered from earliest to latest iteration.
        norm_power (int, optional): Type of norm (e.g. 1 is abs diff, 2 is mean squared difference).
            If less than 1, returns infinity norm (max difference). Defaults to 1.
        divide_by_iters (bool, optional): If True, divide result by $N$.

    Returns:
        np.ndarray: array of $N$ mean differences for each sample over all iterations.
    """
    output_prob_by_iter = np.abs(output_prob_by_iter)
    diff = np.abs(output_prob_by_iter[:-1, :] - output_prob_by_iter[1:, :])
    if norm_power <= 0:
        output_diff = np.max(diff, axis=0)
    else:
        output_diff = (np.sum(diff**norm_power, axis=0) / diff.shape[0])**(1 / norm_power)
    if divide_by_iters:
        output_diff = output_diff / output_prob_by_iter.shape[0]
    return output_diff

def forgetting_events(output_prob_by_iter, batch_mask=None) -> np.ndarray:
    pass #TODO

def create_ordered_batch_mask(train_batch_size, example_start_idx, example_end_idx):
    pass #TODO

def stats_str(array):
    return '<{:0.4f}|{:0.4f}|{:0.4f}>'.format(
        np.min(array), np.mean(array), np.max(array))

class PlotTraining():

    def __init__(self, job, include_examples='all'):
        # filter batch by train/test examples
        self.job = job
        self.include_examples = include_examples
        self.batch_mask = create_ordered_batch_mask(self.job.hparams['batch size'],
                        0, int(self.job.hparams['eval number of train examples']))
        labels = np.array([y for _, y in self.job.get_eval_dataset()])
        self.labels = self._train_eval_filter(labels)

    def _train_eval_filter(self, ndarray):
        split_point = int(self.job.hparams['eval number of train examples'])
        if self.include_examples == 'train':
            return ndarray[:split_point, ...]
        elif self.include_examples == 'test':
            return ndarray[split_point:, ...]
        else:
            return ndarray

    def prob_by_iter(self, replicate):
        base_dir = os.path.join(self.job.save_path, f'model{replicate}')
        for epoch in range(1, self.job.n_epochs):
            logits = torch.load(os.path.join(base_dir, f'eval_logits={epoch}.pt'),
                                map_location=torch.device('cpu'))
            probs = torch.softmax(logits, dim=2)
            for train_batch_idx, prob in enumerate(probs):
                yield self._train_eval_filter(prob), train_batch_idx

    def plot_label_dist(self):
        values, counts = np.unique(self.labels, return_counts=True)
        plt.bar(values, counts)
        self.job.save_obj_to_subdir(plt, 'metrics', self.include_examples + '_label_dist')

    def generate_scores(self, score_fn):
        # reduces output dimensionality from (R x I x N x C) to (R x I x N)
        # by calculating score_fn over C classes
        print(f'Scoring {score_fn.__name__} over replicates...')
        scores = []
        for i in range(self.job.n_replicates):
            start_time = time.perf_counter()
            score = [score_fn(prob, self.labels) for prob, _ in self.prob_by_iter(i)]
            scores.append(np.stack(score, axis=0))
            print(f'm={i} s:{stats_str(scores)} t={time.perf_counter() - start_time}')
        return np.stack(scores, axis=0)

    def generate_metrics(self, scores, metric_fn):
        # reduces output dimensionality from (R x I x N) to (R x N) by
        # calculating metric_fn over I iterations
        name = self.include_examples + '_' + metric_fn.__name__
        print(f'Metric {name} over replicates...')
        metrics = []
        for i, score in enumerate(scores):
            start_time = time.perf_counter()
            metrics.append(metric_fn(score))
            print(f'm={i}, metric:{stats_str(metrics)} t={time.perf_counter() - start_time}')
        return np.stack(metrics, axis=0)

    def plot_score_trajectories(self, dict_scores):
        for name, scores in dict_scores.items():
            for i, replicate in enumerate(scores):
                f = plt.figure()
                f.set_figwidth(16)
                f.set_figheight(8)
                print(f'Plotting trajectories for {name} replicate {i}...')
                start_time = time.perf_counter()
                for example in replicate.transpose():
                    plt.plot(example, linewidth=0.8, alpha=0.1)
                plt.title(name)
                print(f'Plotted in t={time.perf_counter() - start_time}')
                self.job.save_obj_to_subdir(plt, 'metrics',
                    f'training_trajectories_{name}_{i}')

    def plot_metric_rank_qq(self, dict_metrics):
        for name, metrics in dict_metrics.items():
            n_samples = len(metrics[0])
            rank = np.arange(n_samples)
            # plot sorted metrics as lines
            colors = plt.cm.jet(np.linspace(0., 1., n_samples))
            for i, metric in enumerate(metrics):
                plt.plot(rank, np.sort(metric), color=colors[i], alpha=0.4)
            plt.title(name)
            self.job.save_obj_to_subdir(plt, 'metrics', name)

    def metrics_to_ranks(self, metrics):
        # assume dimensions are (replicates, examples)
        n_rep, n_example = metrics.shape
        sorted_idx = np.argsort(metrics, axis=1)
        rep_idx = np.arange(n_rep).reshape(n_rep, 1).repeat(n_example, axis=1)
        rank_idx = np.arange(n_example).reshape(1, n_example).repeat(n_rep, axis=0)
        ranks = np.empty_like(sorted_idx)
        ranks[rep_idx, sorted_idx] = rank_idx
        return ranks

    def plot_metric_array(self, dict_metrics):
        # take dict of {name: metric}, plot every combination of mean(metricA) to metricB
        # also include ranked versions of each metric
        names, orders, metrics = [], [], []
        for name, metric in dict_metrics.items():
            names.append(name)
            metrics.append(metric)
            orders.append(np.mean(metric, axis=0))  # average over replicates
            # rank version
            names.append(name + '_rank')
            rank = self.metrics_to_ranks(metric)
            metrics.append(rank)
            orders.append(np.mean(rank, axis=0))  # average over replicates
        # scatter plot for every combination of order, metric
        n_rep = metrics[0].shape[0]
        n_plt = max(len(names), 2)
        fig, axes = plt.subplots(n_plt, n_plt, figsize=(3 * n_plt, 3 *n_plt))
        for i, (name_row, row) in enumerate(zip(names, axes)):
            for j, (name_col, ax) in enumerate(zip(names, row)):
                if i == 0:
                    ax.set_title(name_col)
                if j == 0:
                    ax.set_ylabel(name_row)
                if i <= j:
                    x_data = orders[i].reshape(1, -1).repeat(n_rep, axis=0)
                    y_data = metrics[j]
                    ax.set_xlabel('mean_' + name_col)
                else:
                    x_data = metrics[i]
                    y_data = metrics[j]
                ax.scatter(x_data.flatten(), y_data.flatten(), s=4, alpha=0.2)
        self.job.save_obj_to_subdir(plt, 'metrics',
            self.include_examples + '_metric_rank_' + '-'.join(names))

    def metric_rank_corr_array(self, dict_metrics):
        # take dict of {name: metric}
        # do pairwise rank correlation between two metrics for each replicate
        # if between the same metric and itself, find correlation between replicates
        # also include rank correlation with mean metrics
        names, metrics = [], []
        for name, metric in dict_metrics.items():
            names.append(name)
            metrics.append(metric)
            # mean metrics over replicates
            names.append(name + '_mean')
            metrics.append(np.mean(metric, axis=0))
        n_plt = max(len(names), 2)
        fig, axes = plt.subplots(n_plt, n_plt, figsize=(3 * n_plt, 3 *n_plt))
        for i, (name1, metric1, row) in enumerate(zip(names, metrics, axes)):
            for j, (name2, metric2, ax) in enumerate(zip(names, metrics, row)):
                if i == j:  # do pairwise over same metric, multiple replicates
                    # spearmanr returns (rho, p-value), ignore the p-value
                    correlations = [spearmanr(a, b)[0] for a, b in zip(metric1[:-1], metric1[1:])]
                else:  # do pairwise rank corr between two metrics, per replicate
                    correlations = [spearmanr(a, b)[0] for a, b in zip(metric1, metric2)]
                # plot rank correlations as box plot
                ax.boxplot(correlations)
                if i == 0:
                    ax.set_title(name2)
                if j == 0:
                    ax.set_ylabel(name1)
                ax.set_ylim(0., 1.)
                # also plot individual correlations and p-values as scatter with jitter
                jitter = np.random.normal(1, 0.05, len(correlations))
                ax.plot(jitter, correlations, '.', alpha=0.4)
        self.job.save_obj_to_subdir(plt, 'metrics',
            self.include_examples + '_metric_rho_corr_' + '-'.join(names))

    def batch_forgetting(self, output_prob_by_iter):  # as implemented by Toneva, same as Nikhil'ss
        return forgetting_events(output_prob_by_iter, batch_mask=self.batch_mask)

    def plot_forget_auc_diff_ranks(self):
        self.plot_label_dist()
        # score derived from output probabilities
        correctness = self.generate_scores(outputs_to_correctness)
        self.plot_score_trajectories({'correctness': correctness})
        # metrics derived from scores
        auc = self.generate_metrics(correctness, top_1_auc)
        diff = self.generate_metrics(correctness, diff_norm)
        # TODO forgetting
        # forgetting = self.generate_metrics(correctness, forgetting_events)
        # batch_forgetting = self.generate_metrics(correctness, self.batch_forgetting)
        self.plot_metric_rank_qq({'auc': auc, 'diff': diff})
        self.metric_rank_corr_array({'auc': auc, 'diff': diff})
        self.plot_metric_array({'auc': auc, 'diff': diff})
