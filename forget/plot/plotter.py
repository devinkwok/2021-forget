import numpy as np
from scipy.stats import spearmanr
import matplotlib
import matplotlib.pyplot as plt
from forget.job import Job

from forget.metrics.transforms import jaccard_similarity, moving_average


class Plotter:
    def __init__(self, job: Job, subdir):
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
        self,
        plt_obj,
        row_metric,
        col_metric,
        mask,
        group,
        ylim=None,
        smoothing=1,
        hlines=None,
        vlines=None,
    ):  # ignore row_metric
        if ylim is None:
            ylim = [min(0, np.min(col_metric)), max(1.0, np.max(col_metric))]
        plt_obj.set_ylim(*ylim)
        last_dim = col_metric.shape[-1]
        col_metric = col_metric.reshape(-1, last_dim)
        group_idx = np.unique(group)
        colors = self.cmap(self.normalize_colors(group_idx))
        max_x, min_y, max_y = 0, 0, 1
        for g, c in zip(group_idx, colors):
            group_mask = np.logical_and(mask, group == g).reshape(-1, last_dim)
            for data, m in zip(col_metric, group_mask):
                min_y = min(min_y, np.min(data))
                max_y = max(max_y, np.max(data))
                max_x = max(max_x, len(data[m]))
                line = data[m]
                if smoothing > 1:
                    smoothed_data = moving_average(
                        line.reshape(-1, 1), smoothing, pad_ends=True
                    ).flatten()
                    plt_obj.plot(line, linewidth=0.6, color=c, alpha=0.2)
                    plt_obj.plot(smoothed_data, linewidth=0.9, color=c, alpha=0.5)
                else:
                    plt_obj.plot(line, linewidth=0.7, color=c, alpha=0.4)
        if hlines is not None:
            plt_obj.hlines(hlines, 0, max_x, colors="black", linestyles="dotted")
        if vlines is not None:
            plt_obj.vlines(vlines, min_y, max_y, colors="black", linestyles="dotted")

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
            linewidth=0,
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
            linewidth=0,
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
        height=None,
        width=None,
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
        if width is None:
            width = max(5, 3 * cols)
        if height is None:
            height = max(5, 3 * rows)
        fig, axes = plt.subplots(rows, cols, figsize=(width, height))
        if rows == 1:
            axes = [axes]
        if cols == 1:
            axes = [[x] for x in axes]
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

    def plot_smooth_curves(
        self,
        name,
        metrics,
        group=None,
        group_names=None,
        mask=None,
        smoothing=1,
        hlines=None,
        vlines=None,
    ):
        def plt_smooth_curve(ax, row_metric, col_metric, mask=None, group=None):
            self.plt_curves(
                ax,
                row_metric,
                col_metric,
                smoothing=smoothing,
                mask=mask,
                group=group,
                hlines=hlines,
                vlines=vlines,
            )

        self.plot_array(
            plt_smooth_curve,
            name,
            row_metrics=metrics,
            mask=mask,
            group=group,
            group_names=group_names,
            width=20,
        )

    def plot_corr_heatmap(
        self,
        prefix,
        row_metrics,
        col_metrics,
        group,
        group_names,
        mask=None,
        width=None,
        height=10,
        average_corr_over_reps=1,
    ):
        diverging_cmap = plt.get_cmap("RdBu")
        row_labels = list(row_metrics.keys())
        col_labels = list(col_metrics.keys())
        group_idx = np.unique(group)
        if width is None:
            width = max(10, 10 * len(group_idx))
        _, axes = plt.subplots(2, len(group_idx), figsize=(width, height))
        if len(group_idx) == 1:
            axes = [axes]
        if mask is None:
            mask = np.ones_like(group, dtype=bool)
        for g, name, ax, extra_ax in zip(group_idx, group_names, axes[0], axes[1]):
            group_mask = np.logical_and(mask, group == g).flatten()
            corr_array = np.empty([len(row_metrics), len(col_metrics)])
            for i, rv in enumerate(row_metrics.values()):
                for j, cv in enumerate(col_metrics.values()):
                    corr_array[i, j] = spearmanr(
                        rv.flatten()[group_mask], cv.flatten()[group_mask]
                    )[0]
            # if set, average over groups of metrics
            x_labels = col_labels
            y_labels = row_labels
            if average_corr_over_reps > 1:
                (
                    corr_array,
                    x_labels,
                    y_labels,
                    correlations,
                ) = self.average_corr_matrix_over_n(
                    corr_array, average_corr_over_reps, col_labels, row_labels
                )
                self.boxplot_corr(extra_ax, correlations)
            image = ax.imshow(
                corr_array,
                norm=matplotlib.colors.Normalize(vmin=-1, vmax=1),
                cmap=diverging_cmap,
            )
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_xticklabels(x_labels)
            ax.set_yticklabels(y_labels)
            ax.figure.colorbar(image, ax=ax)
            ax.set_title(name)
            plt.setp(
                ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor"
            )
            plt.subplots_adjust(left=0.2, bottom=0.2)
            for i in range(len(x_labels)):
                for j in range(len(y_labels)):
                    ax.text(
                        j,
                        i,
                        f"{corr_array[i, j]:0.2f}",
                        ha="center",
                        va="center",
                        color="w",
                    )
        self.job.save_obj_to_subdir(plt, self.subdir, f"{prefix}-corr_heatmap")

    def average_corr_matrix_over_n(
        self, corr_array: np.ndarray, n: int, col_labels, row_labels
    ):
        assert (
            len(corr_array.shape) == 2
            and corr_array.shape[0] == corr_array.shape[1]
            and corr_array.shape[0] % n == 0
        ), corr_array.shape
        size = corr_array.shape[0] // n
        output = np.empty([size, size])
        correlations = []
        for i in range(size):
            for j in range(size):
                corr = corr_array[i * n : (i + 1) * n, j * n : (j + 1) * n]
                # if i == j:
                # only use triangular matrix to keep same shape across all
                correlations.append(corr[np.triu_indices(n, k=1)])
                output[i, j] = np.mean(correlations[-1])
        x_labels = [col_labels[i * n] for i in range(size)]
        y_labels = [row_labels[i * n] for i in range(size)]
        return output, x_labels, y_labels, np.stack(correlations, axis=0)

    def plot_bar(self, prefix, metrics):
        labels = list(metrics.keys())
        x = np.arange(len(labels))
        y = [metrics[x] for x in labels]
        _, ax = plt.subplots(1, 1, figsize=(len(labels) * 2, 10))
        ax.bar(x, y)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
        plt.subplots_adjust(bottom=0.3)
        self.job.save_obj_to_subdir(plt, self.subdir, f"{prefix}-bar")
