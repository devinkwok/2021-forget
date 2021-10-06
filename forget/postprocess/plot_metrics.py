import os
import torch
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


class PlotMetrics():

    def __init__(self, job):
        self.job = job

    def _color(self, i, n):
        return plt.cm.jet(i / n)

    def plot_class_counts(self, name, labels):
        values, counts = np.unique(labels, return_counts=True)
        plt.bar(values, counts)
        self.job.save_obj_to_subdir(plt, 'plot-metrics', f'counts_{name}')

    def plot_score_curves(self, name, scores):
        # scores is (G x L x I)
        # G is groups of same colored lines, L is lines, I is iterations (y-values)
        f = plt.figure()
        f.set_figwidth(16)
        f.set_figheight(8)
        plt.ylim(-1., 1.)
        plt.title(name)
        # dotted line at 0
        plt.hlines(0., 0, len(scores), colors='black', linestyles='dotted')
        for i, replicate in enumerate(scores):
            color = self._color(i, len(scores))
            for example in replicate:
                plt.plot(example, linewidth=1., color=color, alpha=0.2)
        self.job.save_obj_to_subdir(plt, 'plot-metrics',
            f'curve_{name}')

    def plot_metric_rank_qq(self, dict_metrics):
        for name, metrics in dict_metrics.items():
            n_samples = len(metrics[0])
            rank = np.arange(n_samples)
            # plot sorted metrics as lines
            colors = plt.cm.jet(np.linspace(0., 1., n_samples))
            for i, metric in enumerate(metrics):
                plt.plot(rank, np.sort(metric), color=colors[i], alpha=0.4)
            plt.title(name)
            self.job.save_obj_to_subdir(plt, 'plot-metrics', name)

    def metrics_to_ranks(self, metrics):
        # assume dimensions are (replicates, examples)
        n_rep, n_example = metrics.shape
        sorted_idx = np.argsort(metrics, axis=1)
        rep_idx = np.arange(n_rep).reshape(n_rep, 1).repeat(n_example, axis=1)
        rank_idx = np.arange(n_example).reshape(1, n_example).repeat(n_rep, axis=0)
        ranks = np.empty_like(sorted_idx)
        ranks[rep_idx, sorted_idx] = rank_idx
        return ranks

    def plot_metric_scatter_array(self, suffix, dict_metrics):
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
                ax.scatter(x_data.flatten(), y_data.flatten(), marker='.', s=4, alpha=0.02)
        self.job.save_obj_to_subdir(plt, 'plot-metrics',
            f'rank{suffix}')

    def plot_metric_rank_corr_array(self, suffix, dict_metrics):
        # do pairwise rank correlation between two metrics for each replicate
        # if between the same metric and itself, find correlation between replicates
        # also include rank correlation with mean metrics
        names, metrics = [], []
        for name, metric in dict_metrics.items():
            names.append(name)
            metrics.append(metric)
            n_rep = metric.shape[0]
            # mean metrics over replicates
            names.append(name + '_mean')
            metrics.append(np.mean(metric, axis=0).reshape(1, -1).repeat(n_rep, axis=0))
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
        self.job.save_obj_to_subdir(plt, 'plot-metrics',
            f'corr{suffix}')

    def plot_metrics(self, dict_metrics, suffix, filter):
        # take dict of {name: metric}
        # each metric is (R x N), where R is replicates, N is examples
        # suffix is added to end of name
        # filter is applied to each metric
        metrics = {k + suffix: filter(v) for k, v in dict_metrics.items()}
        self.plot_metric_rank_qq(metrics)
        self.plot_metric_scatter_array(suffix, metrics)
        self.plot_metric_rank_corr_array(suffix, metrics)
