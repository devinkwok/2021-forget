import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from forget.postprocess.transforms import jaccard_similarity


class PlotMetrics:
    def __init__(self, job, subdir):
        self.job = job
        self.subdir = subdir
        n_train = int(self.job.hparams["eval number of train examples"])
        n_test = int(self.job.hparams["eval number of test examples"])
        self.colors = np.concatenate([np.zeros(n_train), np.ones(n_test)], axis=0)
        self.cmap = plt.cm.viridis

    def _color(self, i, n):
        return self.cmap(i / n)

    def plot_class_counts(self, name, labels):
        values, counts = np.unique(labels, return_counts=True)
        plt.bar(values, counts)
        self.job.save_obj_to_subdir(plt, self.subdir, f"counts_{name}")

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
        plt.hlines(0.0, 0, groups.shape[-1], colors="black", linestyles="dotted")
        for i, lines in enumerate(groups):
            color = self._color(i, len(groups))
            for line in lines:
                plt.plot(line, linewidth=1.0, color=color, alpha=0.2)
        self.job.save_obj_to_subdir(plt, self.subdir, f"curve_{name}")

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
            selected = np.concatenate(
                [
                    scores[:, :, ranks[:n_rank]],
                    scores[:, :, ranks[mid : mid + n_rank]],
                    scores[:, :, ranks[-n_rank:]],
                ],
                axis=2,
            )
            selected = selected[:n_rep, :, :]  # only include n_rep curves
            # reshape from (R, I, N) to (N, R, I) so that colors indicate rank
            self.plot_curves(
                f"{name}-top{n_rank}", selected.transpose(2, 0, 1), ylim=(-1.0, 1.0)
            )

    def plot_metric_rank_qq(self, dict_metrics):
        for name, metric in dict_metrics.items():
            # (R x S x N) or (R x N), make shape consistent
            while len(metric.shape) < 3:
                metric = np.expand_dims(metric, axis=0)
            metric = metric.reshape(-1, metric.shape[-2], metric.shape[-1])
            # sort along N
            self.plot_curves(f"qq_{name}", np.sort(metric, axis=-1))

    def boxplot_corr(self, plt_obj, correlations):
        plt_obj.boxplot(correlations)
        plt_obj.set_ylim(0.0, 1.0)
        # also plot individual correlations and p-values as scatter with jitter
        jitter = np.random.normal(1, 0.05, len(correlations))
        plt_obj.plot(jitter, correlations, ".", alpha=0.4)

    def _plt_corr(self, plt_obj, row_metric, col_metric, on_adjacent_pairs, mask):
        # broadcast to make sure both metrics have same sized R
        m1, m2 = np.broadcast_arrays(row_metric, col_metric)
        if mask is None:
            mask = np.ones_like(m1, dtype=bool)
        assert len(m1.shape) == 2 and len(m2.shape) == 2
        if on_adjacent_pairs:
            m1 = m1[:-1, ...]
            m2 = m2[1:, ...]
        correlations = []
        for a, b, m in zip(m1, m2, mask):  # iterate over first dim R
            correlations.append(spearmanr(a[m], b[m])[0])
        correlations = np.square(np.array(correlations))
        self.boxplot_corr(plt_obj, correlations)

    def plt_self_corr(self, plt_obj, row_metric, col_metric, mask):
        self._plt_corr(plt_obj, row_metric, col_metric, True, mask)

    def plt_pair_corr(self, plt_obj, row_metric, col_metric, mask):
        self._plt_corr(plt_obj, row_metric, col_metric, False, mask)

    def plt_scatter(self, plt_obj, row_metric, col_metric, mask):
        x_data, y_data, colors = np.broadcast_arrays(
            col_metric, row_metric, self.colors
        )
        if mask is None:
            mask = np.ones_like(x_data, dtype=bool)
        for i, (a, b, m, c) in enumerate(zip(x_data, y_data, mask, colors)):
            plt_obj.scatter(
                a[m], b[m], c=c[m], marker=".", s=4, alpha=0.05, cmap=self.cmap
            )
            # plt_obj.scatter(a[m], b[m], c=self._color(i, len(x_data)), marker=".", s=4, alpha=0.05, cmap=self.cmap)

    def plt_jaccard_curve(self, plt_obj, row_metric, col_metric, mask):
        x_data, y_data = np.broadcast_arrays(col_metric, row_metric)
        if mask is None:
            mask = np.ones_like(x_data, dtype=bool)
        for i, (a, b, m) in enumerate(zip(x_data, y_data, mask)):
            x = np.linspace(0, 1.0, len(a[m]))  # normalize x axis from 0 to 100%
            # plot experimental baseline
            rand_y, rand_1, rand_2 = jaccard_similarity(
                np.random.permutation(a[m]), np.random.permutation(a[m])
            )
            plt_obj.plot(
                x, rand_y, color="black", linewidth=1.0, linestyle="dotted", alpha=0.4
            )
            # plot theoretical baseline
            plt_obj.plot(
                x,
                x / (2 - x),
                color="red",
                linewidth=1.0,
                linestyle="dotted",
                alpha=0.4,
            )
            # plot actual curve
            y, x_1, x_2 = jaccard_similarity(a[m], b[m])
            plt_obj.plot(
                x, y, color=self._color(i, len(x_data)), linewidth=1.0, alpha=0.4
            )

    def plt_hist(self, plt_obj, row_metric, col_metric, mask):  # don't use row_metric
        if mask is None:
            mask = np.ones_like(col_metric, dtype=bool)
        dist = col_metric[mask]
        plt_obj.hist(
            dist, bins="rice", density=False, range=(np.min(dist), np.max(dist))
        )

    def plot_array(self, plt_fn, suffix, dict_metrics_row, dict_metrics_col, mask=None):
        # take dict of {name: metric}, plot every combination of mean(metricA) to metricB
        # scatter plot for every combination of dict_metrics_row, dict_metrics_col
        rows, cols = max(len(dict_metrics_row), 2), max(len(dict_metrics_col), 2)
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        for i, (name_row, row) in enumerate(zip(dict_metrics_row.keys(), axes)):
            for j, (name_col, ax) in enumerate(zip(dict_metrics_col.keys(), row)):
                if i == 0:
                    ax.set_title(name_col)
                if j == 0:
                    ax.set_ylabel(name_row)
                plt_fn(ax, dict_metrics_row[name_row], dict_metrics_col[name_col], mask)
        self.job.save_obj_to_subdir(plt, self.subdir, f"{plt_fn.__name__}-{suffix}")
