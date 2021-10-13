import os
import torch
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


class PlotMetrics():

    def __init__(self, job, subdir='plot-metrics'):
        self.job = job
        self.subdir = subdir

    def _color(self, i, n):
        return plt.cm.viridis(i / n)

    def plot_class_counts(self, name, labels):
        values, counts = np.unique(labels, return_counts=True)
        plt.bar(values, counts)
        self.job.save_obj_to_subdir(plt, 'plot-metrics', f'counts_{name}')

    def plot_curves(self, name, groups, ylim=None):
        # groups is (G x L x I)
        # G is groups of same colored lines, L is lines, I is iterations (y-values)
        f = plt.figure()
        f.set_figwidth(16)
        f.set_figheight(8)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.title(name)
        # dotted line at 0
        plt.hlines(0., 0, groups.shape[-1], colors='black', linestyles='dotted')
        for i, lines in enumerate(groups):
            color = self._color(i, len(groups))
            for line in lines:
                plt.plot(line, linewidth=1., color=color, alpha=0.2)
        self.job.save_obj_to_subdir(plt, 'plot-metrics',
            f'curve_{name}')

    def plot_curves_by_rank(self, scores, dict_metrics, n_rank=1, n_rep=3):
        # for each metric, plot the largest and smallest ranking example
        for name, metric in dict_metrics.items():
            # group dims to (RS... N)
            metric = metric.reshape(-1, metric.shape[-1])
            scores = scores.reshape(-1, scores.shape[-2], scores.shape[-1])
            # take average over remaining axes to rank examples
            ranks = np.argsort(np.mean(metric, axis=0))
            mid = metric.shape[-1] // 2 - n_rank // 2
            # combine groups of worst, middle, best scores
            selected = np.concatenate([
                    scores[:, :, ranks[:n_rank]],
                    scores[:, :, ranks[mid:mid+n_rank]],
                    scores[:, :, ranks[-n_rank:]]
                ], axis=2)
            selected = selected[:n_rep, :, :]  # only include n_rep curves
            # reshape from (R, I, N) to (N, R, I) so that colors indicate rank
            self.plot_curves(f'{name}-top{n_rank}', selected.transpose(2, 0, 1), ylim=(-1., 1.))

    def plot_metric_rank_qq(self, dict_metrics):
        for name, metric in dict_metrics.items():
            # (R x S x N) or (R x N), make shape consistent
            while len(metric.shape) < 3:
                metric = np.expand_dims(metric, axis=0)
            metric = metric.reshape(-1, metric.shape[-2], metric.shape[-1])
            # sort along N
            self.plot_curves(f'qq_{name}', np.sort(metric, axis=-1))

    def metrics_to_ranks(self, metrics):
        # rank along last dimension
        sorted_idx = np.argsort(metrics, axis=-1)
        rank_idx = np.arange(metrics.shape[-1])
        ranks = np.empty_like(sorted_idx)
        np.put_along_axis(ranks, sorted_idx, rank_idx, axis=-1)
        return ranks

    def plot_metric_scatter_array(self, suffix, dict_metrics):
        # take dict of {name: metric}, plot every combination of mean(metricA) to metricB
        # also include ranked versions of each metric
        # assume each metric has dimensions (..., R, N)
        # take mean over R to get x-values, individual N for y-values
        names, metrics, means = [], [], []
        for name, metric in dict_metrics.items():
            names.append(name)
            metrics.append(metric)
            # average over R in a broadcastable way
            mean_metric = np.mean(metric, axis=-2, keepdims=True)
            means.append(mean_metric)
            # rank version of metric
            names.append(name + '_rk')
            ranks = self.metrics_to_ranks(metric)
            metrics.append(ranks)
            # mean of ranks, don't use rank of means, i.e.
            # means.append(self.metrics_to_ranks(mean_metric))
            # since if reps have different ranges their mean will be biased
            # whereas ranks have fixed scale
            means.append(np.mean(ranks, axis=-2, keepdims=True))
        # scatter plot for every combination of order, metric
        n_plt = len(names)
        fig, axes = plt.subplots(n_plt, n_plt, figsize=(3 * n_plt, 3 *n_plt))
        for i, (name_row, row) in enumerate(zip(names, axes)):
            for j, (name_col, ax) in enumerate(zip(names, row)):
                if i == 0:
                    ax.set_title(name_col)
                if j == 0:
                    ax.set_ylabel(name_row)
                if i <= j:
                    x_data = means[j]
                    y_data = metrics[i]
                    ax.set_xlabel('mean')
                else:
                    x_data = metrics[j]
                    y_data = metrics[i]
                print(f'\tbroadcasting x={x_data.shape}, y={y_data.shape}')
                x_data, y_data = np.broadcast_arrays(x_data, y_data)
                ax.scatter(x_data.flatten(), y_data.flatten(), marker='.', s=4, alpha=0.02)
        self.job.save_obj_to_subdir(plt, 'plot-metrics',
            f'rank{suffix}')

    def plot_metric_rank_corr_array(self, suffix, dict_metrics):
        # do pairwise rank correlation between two metrics for each replicate
        # if between the same metric and itself, find correlation between replicates
        # also include rank correlation with mean metrics
        names, metrics = [], []
        for name, metric in dict_metrics.items():
            # reshape to (..., R, N), 
            metric = metric.reshape(-1, metric.shape[-2], metric.shape[-1])
            names.append(name)
            metrics.append(metric)
            # mean metrics over replicates
            names.append(name + '_mu')
            metrics.append(np.mean(metric, axis=-2, keepdims=True))
        n_plt = len(names)
        fig, axes = plt.subplots(n_plt, n_plt, figsize=(3 * n_plt, 3 *n_plt))
        for i, (name_row, metric_row, row) in enumerate(zip(names, metrics, axes)):
            for j, (name_col, metric_col, ax) in enumerate(zip(names, metrics, row)):
                # broadcast so that if R is missing, the same metric is repeated over R
                # if additional dims present, repeat the correlations over these dims
                # e.g. (R, N), (S, R, N) -> for each R, correlate (R, N) and (..., N) over all S
                # e.g. (N), (S, R, N) -> correlate (N) over all R and all S
                # e.g. (S, R, N) with itself -> correlate r_n with r_{n+1} for 1...N, then repeat for all S
                correlations = []
                if i == j:  # enumerate over R and do pairwise for S*(R-1) corr
                    # spearmanr returns (rho, p-value), ignore the p-value
                    for metric in metric_row:  # iterate over S
                        correlations.append(np.array(  # iterate over R
                            [spearmanr(a, b)[0] for a, b in zip(metric[:-1], metric[1:])]))
                else:  # do pairwise between two metrics over each of R for S*R corr
                    metric_row, metric_col = np.broadcast_arrays(metric_row, metric_col)
                    print(f'\tbroadcasting a={metric_row.shape}, b={metric_col.shape}')
                    for m1, m2 in zip(metric_row, metric_col):  # iterate over S
                        correlations.append(np.array([  # iterate over R
                            spearmanr(a, b)[0] for a, b in zip(m1, m2)]))
                # plot squared values so they don't fall below 0
                correlations = np.square(np.concatenate(correlations, axis=0))
                # plot rank correlations as box plot
                ax.boxplot(correlations)
                if i == 0:
                    ax.set_title(name_col)
                if j == 0:
                    ax.set_ylabel(name_row)
                ax.set_ylim(0., 1.)
                # also plot individual correlations and p-values as scatter with jitter
                jitter = np.random.normal(1, 0.05, len(correlations))
                ax.plot(jitter, correlations, '.', alpha=0.4)
        self.job.save_obj_to_subdir(plt, 'plot-metrics',
            f'corr{suffix}')
