import numpy as np
import matplotlib.pyplot as plt
from forget.metrics.metrics import Metrics
from forget.metrics.transforms import stats_str
from forget.plot.plotter import Plotter


class DiffPlots(Plotter):
    def __init__(self, parser, filter_list) -> None:
        self.parser = parser
        first_job = self.parser.list_jobs()[0]
        super().__init__(first_job, f"cross_job_plots-ep{first_job.n_epochs}")
        self.train_test_groups = np.concatenate(
            [
                np.zeros(first_job.n_logit_train_examples),
                np.ones(first_job.n_logit_test_examples),
            ],
            axis=0,
        )
        self.filter_list = filter_list

    def load_metrics(self, job):
        metrics_dict = Metrics(job, self).load_metrics()
        metrics_dict = {n: metrics_dict[n] for n in self.filter_list}
        metrics_dict = Metrics.average_over_samples(metrics_dict)
        return metrics_dict

    def mean(self, metrics):
        return Metrics.apply(metrics, lambda metric: np.mean(metric, axis=0))

    def std(self, metrics):
        return Metrics.apply(metrics, lambda metric: np.std(metric, axis=0))

    def apply_pair(self, dict_metrics_1, dict_metrics_2, transform_fn, suffix=""):
        if suffix != "":
            suffix = "-" + suffix
        output_metrics = {}
        for name, metric_1 in dict_metrics_1.items():
            metric_2 = dict_metrics_2[name]
            output_metrics[f"{name}{suffix}"] = transform_fn(metric_1, metric_2)
        return output_metrics

    def mean_pair(self, metrics_1, metrics_2):
        return self.apply_pair(metrics_1, metrics_2, lambda m1, m2: (m1 + m2) * 0.5)

    def diff(self, metrics_1, metrics_2):
        return self.apply_pair(metrics_1, metrics_2, lambda m1, m2: m2 - m1)

    def plot_diff_and_variance(self, job_1, job_2):
        metrics_1 = self.load_metrics(job_1)
        metrics_2 = self.load_metrics(job_2)
        # get mean per example over replicates
        mean_1 = self.mean(metrics_1)
        mean_2 = self.mean(metrics_2)
        # get variance per example over replicates
        std_1 = self.std(metrics_1)
        std_2 = self.std(metrics_2)
        pooled_var = self.mean_pair(std_1, std_2)  # pool variance
        diff = self.diff(mean_1, mean_2)
        # scatter diff of means and variance per example
        rows = len(metrics_1)
        height = max(5, 5 * rows)
        fig, axes = plt.subplots(rows, 1, figsize=(8, height))
        for ax, (name, mean) in zip(axes, mean_1.items()):
            y = diff[name]
            mean_y = np.mean(y)
            err = pooled_var[name]
            colors = self.cmap(self.normalize_colors(self.train_test_groups))
            ax.set_title(name)
            ax.set_ylabel("Difference of means")
            ax.set_xlabel("Mean value over replicates")
            ax.scatter(
                mean,
                y,
                c=colors,
                alpha=0.4,
                marker=".",
                linewidth=0,
            )
            ratio = np.clip((y - mean_y) / (err + 1e-9), -5.0, 5.0)
            significance_colors = plt.get_cmap("RdBu")(self.normalize_colors(ratio))
            ax.hlines(
                mean_y, np.min(mean), np.max(mean), colors="black", linestyles="dotted"
            )
            ax.vlines(mean, y - err, y + err, colors=significance_colors, alpha=0.3)
        self.job.save_obj_to_subdir(
            plt, self.subdir, f"{job_1.name}-{job_2.name}-diff-scatter"
        )

        fig, axes = plt.subplots(rows, 1, figsize=(8, height))
        for ax, (name, mean) in zip(axes, mean_1.items()):
            x = mean
            y = mean_2[name]
            mean_diff = np.mean(y - x)
            err_x = std_1[name]
            err_y = std_2[name]
            colors = self.cmap(self.normalize_colors(self.train_test_groups))
            ax.set_title(name)
            ratio = np.clip((y - x - mean_diff) / (err_x + err_y + 1e-9), -5.0, 5.0)
            c = plt.get_cmap("RdBu")(self.normalize_colors(ratio))
            print(
                "x",
                stats_str(x),
                "y",
                stats_str(y),
                "std_x",
                stats_str(err_x),
                "std_y",
                stats_str(err_y),
                "ratio",
                stats_str(ratio),
            )
            # ax.hlines(y, x - err_x, x + err_x, colors=c, alpha=0.15)
            # ax.vlines(x, y - err_y, y + err_y, colors=c, alpha=0.15)
            ax.axline((0, 0), slope=1, color="black", linestyle="dotted")
            ax.scatter(
                x,
                y,
                c=colors,
                alpha=0.2,
                marker=".",
                linewidth=0,
            )
        self.job.save_obj_to_subdir(
            plt, self.subdir, f"{job_1.name}-{job_2.name}-means-scatter"
        )
