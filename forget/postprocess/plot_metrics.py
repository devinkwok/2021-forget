import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from forget.postprocess.transforms import jaccard_similarity


class PlotMetrics:
    def __init__(self, job, subdir):
        self.job = job
        self.subdir = subdir
        self.cmap = plt.get_cmap("hsv")

    def normalize_colors(self, colors):
        # add 1 to avoid div by 0 and prevent overlap in circular color maps
        return (colors - np.min(colors)) / (np.max(colors) - np.min(colors) + 1)

    def plot_class_counts(self, name, labels):
        values, counts = np.unique(labels, return_counts=True)
        plt.bar(values, counts)
        self.job.save_obj_to_subdir(plt, self.subdir, f"counts_{name}")

    def plt_curves(
        self, plt_obj, row_metric, col_metric, mask, group, ylim=None
    ):  # ignore row_metric
        if ylim is None:
            ylim = [min(0, np.min(col_metric)), max(1.0, np.max(col_metric))]
        plt_obj.set_ylim(*ylim)
        last_dim = col_metric.shape[-1]
        col_metric = col_metric.reshape(-1, last_dim)
        group_idx = np.unique(group)
        colors = self.cmap(self.normalize_colors(group_idx))
        max_x = 0
        for g, c in zip(group_idx, colors):
            group_mask = np.logical_and(mask, group == g).reshape(-1, last_dim)
            for data, m in zip(col_metric, group_mask):
                max_x = max(max_x, len(data[m]))
                plt_obj.plot(data[m], linewidth=1.0, color=c, alpha=0.2)
        # dotted line at 0
        plt_obj.hlines(0.0, 0, max_x, colors="black", linestyles="dotted")

    def plt_quantiles(
        self, plt_obj, row_metric, col_metric, mask, group
    ):  # ignore row_metric
        sorted_idx = np.argsort(col_metric, axis=-1)
        col_metric = np.take_along_axis(col_metric, sorted_idx, axis=-1)
        mask = np.take_along_axis(mask, sorted_idx, axis=-1)
        group = np.take_along_axis(group, sorted_idx, axis=-1)
        self.plt_curves(plt_obj, row_metric, col_metric, mask, group)

    def boxplot_corr(self, plt_obj, correlations):
        assert len(correlations.shape) == 2  # (group, correlation)
        plt_obj.boxplot(correlations.T)  # transpose to (correlation, group)
        plt_obj.set_ylim(0.0, 1.0)
        # also plot individual correlations as scatter with jitter
        jitter = np.random.normal(1, 0.05, correlations.shape).flatten()
        group_idx = np.arange(correlations.shape[0]).repeat(correlations.shape[1])
        plt_obj.scatter(
            group_idx + jitter,
            correlations.flatten(),
            s=8,
            alpha=0.4,
            marker=".",
            c=self.cmap(self.normalize_colors(group_idx)),
        )

    def _plt_corr(
        self, plt_obj, row_metric, col_metric, on_adjacent_pairs, mask, group
    ):
        if on_adjacent_pairs:  # draw from different samples along 2nd last dim
            assert row_metric.shape[-2] > 1, row_metric.shape
            row_metric = row_metric[..., :-1, :]
            col_metric = col_metric[..., 1:, :]
        last_dim = row_metric.shape[-1]
        row_metric = row_metric.reshape(-1, last_dim)
        col_metric = col_metric.reshape(-1, last_dim)
        correlations = []
        for g in np.unique(group):
            corr_per_group = []
            group_mask = np.logical_and(mask, group == g).reshape(-1, last_dim)
            for a, b, m in zip(row_metric, col_metric, group_mask):
                corr_per_group.append(spearmanr(a[m], b[m])[0])
            correlations.append(corr_per_group)
        self.boxplot_corr(plt_obj, np.square(np.array(correlations)))

    def plt_self_corr(self, plt_obj, row_metric, col_metric, mask, group):
        self._plt_corr(plt_obj, row_metric, col_metric, True, mask, group)

    def plt_pair_corr(self, plt_obj, row_metric, col_metric, mask, group):
        self._plt_corr(plt_obj, row_metric, col_metric, False, mask, group)

    def plt_scatter(self, plt_obj, row_metric, col_metric, mask, group):
        # normalize group values between 0, 1 to make colors
        color = self.normalize_colors(group)
        plt_obj.scatter(
            col_metric[mask],
            row_metric[mask],
            c=self.cmap(color[mask]),
            s=4,
            alpha=0.2,
            marker=".",
        )

    def plt_jaccard_curve(self, plt_obj, row_metric, col_metric, mask, group):
        self._jaccard_curve(
            plt_obj, row_metric, col_metric, mask, group, on_adjacent_pairs=False
        )

    def plt_pair_jaccard(self, plt_obj, row_metric, col_metric, mask, group):
        self._jaccard_curve(
            plt_obj, row_metric, col_metric, mask, group, on_adjacent_pairs=True
        )

    def _jaccard_curve(
        self, plt_obj, row_metric, col_metric, mask, group, on_adjacent_pairs=False
    ):
        if on_adjacent_pairs:  # draw from different samples along 2nd last dim
            assert row_metric.shape[-2] > 1, row_metric.shape
            row_metric = row_metric[..., :-1, :]
            col_metric = col_metric[..., 1:, :]
        last_dim = row_metric.shape[-1]
        row_metric = row_metric.reshape(-1, last_dim)
        col_metric = col_metric.reshape(-1, last_dim)
        group_idx = np.unique(group)
        colors = self.cmap(self.normalize_colors(group_idx))
        for g, c in zip(group_idx, colors):
            group_mask = np.logical_and(mask, group == g).reshape(-1, last_dim)
            for x_data, y_data, m in zip(col_metric, row_metric, group_mask):
                x = np.linspace(
                    0, 1.0, len(x_data[m])
                )  # normalize x axis from 0 to 100%
                # plot theoretical baseline
                plt_obj.plot(
                    x,
                    x / (2 - x),
                    color="black",
                    linewidth=1.0,
                    linestyle="dotted",
                    alpha=0.4,
                )
                # plot actual curve
                y, _, _ = jaccard_similarity(x_data[m], y_data[m])
                plt_obj.plot(x, y, color=c, linewidth=1.0, alpha=0.4)

    def plt_hist(
        self, plt_obj, row_metric, col_metric, mask, group
    ):  # ignore row_metric
        group_idx = np.unique(group)
        colors = self.cmap(self.normalize_colors(group_idx))
        for g, c in zip(group_idx, colors):
            group_mask = np.logical_and(mask, group == g)
            dist = col_metric[group_mask]
            plt_obj.hist(
                dist,
                bins="rice",
                density=False,
                range=(np.min(dist), np.max(dist)),
                color=c,
                alpha=0.4,
            )

    def _dummy_variable(self, name, dict_metrics):
        key, value = list(dict_metrics.items())[0]
        return {name: value}

    def plot_array(
        self,
        plt_fn,
        prefix,
        row_metrics=None,
        col_metrics=None,
        mask=None,
        group=None,
        group_names=None,
    ):
        dict_row = row_metrics
        dict_col = col_metrics
        # add dummy variable to allow broadcasting
        if row_metrics is None:
            dict_row = self._dummy_variable(prefix, dict_col)
        if col_metrics is None:
            dict_col = self._dummy_variable(prefix, dict_row)
        # make trivial mask/group to allow broadcasting
        if mask is None:
            mask = np.ones([1], dtype=bool)
        if group is None:
            group = np.ones([1])
        # scatter plot for every combination of dict_row, dict_col
        rows, cols = len(dict_row), len(dict_col)
        fig_size = (max(8, 3 * cols), max(8, 3 * rows))
        fig, axes = plt.subplots(rows, cols, figsize=fig_size)
        if cols == 1:
            axes = [[x] for x in axes]
        if rows == 1:
            axes = [axes]
        for i, (name_row, row) in enumerate(zip(dict_row.keys(), axes)):
            for j, (name_col, ax) in enumerate(zip(dict_col.keys(), row)):
                if i == 0:
                    ax.set_title(name_col)
                if j == 0:
                    ax.set_ylabel(name_row)
                # broadcast to make all equivalent
                row_metric, col_metric, metric_mask, metric_group = np.broadcast_arrays(
                    dict_row[name_row], dict_col[name_col], mask, group
                )
                # send non-none metric only
                if row_metrics is None:
                    row_metric = col_metric
                if col_metrics is None:
                    col_metric = row_metric
                plt_fn(ax, row_metric, col_metric, mask=metric_mask, group=metric_group)
        # make legend assuming discrete group indexes
        if group_names is not None:
            color = self.normalize_colors(np.unique(group))
            assert len(group_names) == len(color)
            dummies = [
                axes[0][0].plot([], [], ls="-", c=self.cmap(c))[0] for c in color
            ]
            fig.legend(dummies, group_names, loc="right")
        shortname = plt_fn.__name__[len("plt_") :]
        self.job.save_obj_to_subdir(plt, self.subdir, f"{prefix}-{shortname}")

    def plot_curves_by_rank(self, scores, dict_metrics, n_rank=1):
        # for each metric, plot largest, middle, and smallest ranked example of first replicate/sample
        scores = scores.reshape(-1, scores.shape[-2], scores.shape[-1])[0]
        mid = scores.shape[-1] // 2 - n_rank // 2  # rank of middle example
        selected = {}
        for name, metric in dict_metrics.items():
            # group dims to (RS... N) and take first element of RS
            metric = metric.reshape(-1, metric.shape[-1])[0]
            assert metric.shape[-1] == scores.shape[-1]
            ranks = np.argsort(metric)
            lines = np.concatenate(
                [
                    scores[:, ranks[:n_rank]],
                    scores[:, ranks[mid : mid + n_rank]],
                    scores[:, ranks[-n_rank:]],
                ],
                axis=1,
            )
            # swap axes to (lines, iters)
            selected[name] = lines.transpose(1, 0)
        group = (
            np.arange(3).repeat(n_rank).reshape(-1, 1).repeat(scores.shape[-2], axis=1)
        )
        self.plot_array(
            self.plt_curves,
            "examples",
            row_metrics=selected,
            col_metrics=None,
            group=group,
            group_names=["Low", "Med", "High"],
        )
