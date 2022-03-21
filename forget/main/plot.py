import os
import datetime
import torch
import numpy as np
from forget.plot.diff_plot import DiffPlots

from forget.plot.job_plot import JobPlots
from forget.main.parser import readConfig
from forget.main.experiment import run_experiment
from forget.metrics.metrics import Metrics
from forget.plot.plotter import Plotter
from forget.metrics.train_metrics import TrainMetrics
from forget.plot.pca import pca_metrics


class ExperimentPlotter:
    def __init__(self) -> None:
        self.filter_type = "first_forget"
        # filter relevant metrics
        self.key_metric_names = [
            f"exnoise_erase_greyscale-{self.filter_type}",
            f"exnoise_gauss_channel-{self.filter_type}",
            f"prune_magn_notrain-{self.filter_type}",
            f"mnoise_mult_conv-{self.filter_type}",
            "train-first_learn",
            "train-forget",
            "eval-accuracy",
            "eval-margin",
            "eval-loss",
        ]
        self.parser = readConfig()
        self.split_percent = 0.5
        self.carlini_metrics = self.get_carlini_metrics()

        self.perturb_sample_plots()
        self.plots_per_job()
        self.plot_curves()
        self.cross_job_plots()
        self.cross_job_heatmap_averaged()
        self.plot_cross_job_diff()
        self.plot_pca()
        print(f"Plots finished at t={datetime.datetime.now()}")

    def perturb_sample_plots(self):
        for job in self.parser.list_jobs():
            if "do plot" in job.hparams:
                perturbations = run_experiment.perturbations(job)
                plotter = Plotter(job, subdir=f"perturb-plots-ep{job.n_epochs}")
                metrics_generator = TrainMetrics(job, plotter)
                job_plots = JobPlots(job, plotter)
                metrics = metrics_generator.load_metrics()
                for perturbation in perturbations:
                    job_plots.plot_metrics_by_sample(metrics, perturbation)

    def get_carlini_metrics(self):
        job = self.parser.list_jobs()[0]
        metrics = TrainMetrics(job, Plotter(job, "")).load_metrics()
        carlini_metrics = Metrics.filter_metrics(
            Metrics.average_over_samples(metrics), "carlini"
        )
        return carlini_metrics

    def get_metrics(self, job):
        plotter = Plotter(job, subdir=f"{self.filter_type}_plots-ep{job.n_epochs}")
        job_plots = JobPlots(
            job, plotter
        )  # TODO remove need for object reference to call static functions
        metrics_generator = TrainMetrics(job, plotter)
        metrics = metrics_generator.load_metrics()

        cutoff = int(job.hparams["plot learned before cutoff"])
        learned_before_iter = (job.n_epochs - cutoff) * job.n_iter_per_epoch
        # summarize noise_metrics over S to get R x N (same as train_metrics)
        averaged_metrics = Metrics.average_over_samples(metrics)
        perturb_metrics = Metrics.filter_metrics(averaged_metrics, self.filter_type)
        train_metrics = Metrics.filter_metrics(averaged_metrics, "train-first_learn")
        train_metrics["train-forget"] = averaged_metrics["train-forget"]
        # only keep one first_below metric
        tvd_keys = [
            "tvd-centerOfMass",
            "tvd-first_below_1_16",
        ]
        tvd_metrics = {k: metrics[k] for k in tvd_keys}
        per_replicate_metrics = {**perturb_metrics, **train_metrics}
        median_metrics, median_mask = job_plots.masked_median(
            per_replicate_metrics, learned_before_iter
        )
        per_replicate_metrics = {**per_replicate_metrics, **self.carlini_metrics}
        median_over_replicates_metrics = {
            **median_metrics,
            **Metrics.average_over_replicates(
                Metrics.average_over_samples(tvd_metrics)
            ),
            **Metrics.average_over_replicates(self.carlini_metrics),
        }
        return per_replicate_metrics, median_over_replicates_metrics, median_mask

    def plots_per_job(self):
        for job in self.parser.list_jobs():
            plotter = Plotter(job, subdir=f"{self.filter_type}_plots-ep{job.n_epochs}")
            if "do plot" in job.hparams:
                job_plots = JobPlots(job, plotter)
                (
                    per_replicate_metrics,
                    median_over_replicates_metrics,
                    median_mask,
                ) = self.get_metrics(job)
                job_plots.plot_pairwise(
                    {"id": per_replicate_metrics},
                    f"train_perturb{np.sum(median_mask)}",
                    mask=median_mask,
                )
                median_mask = np.all(median_mask, axis=0)
                job_plots.plot_pairwise(
                    {"med": median_over_replicates_metrics},
                    f"medmask{np.sum(median_mask)}",
                    mask=median_mask,
                )
                # job_plots.plot_all(per_replicate_metrics)

    def plot_curves(self):
        for job in self.parser.list_jobs():
            plotter = Plotter(job, subdir=f"plots-ep{job.n_epochs}")
            job_plots = JobPlots(job, plotter)
            subdir = os.path.join(job.save_path, Metrics(job, plotter).subdir)
            for file in os.listdir(subdir):
                if file.endswith(".mdict"):
                    curve_dict = torch.load(os.path.join(subdir, file))
                    job_plots.plot_iter_curves(curve_dict, file[: -len(".mdict")])

    def get_cross_job_metrics(self):
        # load metrics from each job
        job_plots = None
        job_metrics = []
        n_replicates = []
        for job in self.parser.list_jobs():
            if job_plots is None:
                plotter = Plotter(job, f"cross_job_plots-ep{job.n_epochs}")
                job_plots = JobPlots(job, plotter)
            # per_replicate_metrics, median_over_replicates_metrics, _ = self.get_metrics(job)
            metrics_dict = Metrics(job, plotter).load_metrics()
            metrics_dict = {n: metrics_dict[n] for n in self.key_metric_names}
            metrics_dict = Metrics.average_over_samples(metrics_dict)
            job_name = f"{job.hparams['model parameters']}_{job.name}"
            job_metrics += [(job_name, metrics_dict)]
            n_replicates.append(job.n_replicates)
        plotter = job_plots.plotter
        group_names = metrics_dict.keys()
        group_idx = np.arange(len(group_names)).reshape(-1, 1)
        group_idx = np.broadcast_to(
            group_idx, (group_idx.shape[0], list(metrics_dict.values())[0].shape[-1])
        )
        min_replicates = min(n_replicates)
        return job_plots, plotter, job_metrics, min_replicates, group_names, group_idx

    def cross_job_iter(self, job_metrics):
        for i in range(len(job_metrics)):
            for j in range(i + 1, len(job_metrics)):
                job1_name, job1_metrics = job_metrics[i]
                job2_name, job2_metrics = job_metrics[j]
                job1_name = job1_name[len("cifar_resnet_") :]
                job2_name = job2_name[len("cifar_resnet_") :]
                yield job1_name, job1_metrics, job2_name, job2_metrics

    def cross_job_plots(self):
        (
            job_plots,
            plotter,
            job_metrics,
            min_replicates,
            group_names,
            group_idx,
        ) = self.get_cross_job_metrics()
        for job1_name, job1_metrics, job2_name, job2_metrics in self.cross_job_iter(
            job_metrics
        ):
            output_prefix = f"{job1_name}-{job2_name}"
            # plot pairs of jobs with average over replicates
            job_plots.scatter(output_prefix, job1_metrics, job2_metrics)
            # plot correlation heatmap over individual replicates
            new_metrics = {}
            for replicate in range(min_replicates):
                stack = [job1_metrics[name][replicate] for name in group_names]
                new_metrics[f"{job1_name}.{replicate}"] = np.stack(stack, axis=0)
            for replicate in range(min_replicates):
                stack = [job2_metrics[name][replicate] for name in group_names]
                new_metrics[f"{job2_name}.{replicate}"] = np.stack(stack, axis=0)
            plotter.plot_corr_heatmap(
                output_prefix,
                new_metrics,
                new_metrics,
                group_idx,
                group_names,
            )

    def cross_job_heatmap_averaged(self):
        (
            _,
            plotter,
            job_metrics,
            min_replicates,
            group_names,
            group_idx,
        ) = self.get_cross_job_metrics()
        metrics_by_replicate = {}
        for i in range(len(job_metrics)):
            job_name, job = job_metrics[i]
            for replicate in range(min_replicates):
                stack = [job[name][replicate] for name in group_names]
                metrics_by_replicate[f"{job_name}.{replicate}"] = np.stack(
                    stack, axis=0
                )
        for k, v in self.carlini_metrics.items():
            split_point = int(self.split_percent * v.shape[-1])
            mask = v > np.sort(v)[0, split_point]
            plotter.plot_corr_heatmap(
                f"{k}{split_point}",
                metrics_by_replicate,
                metrics_by_replicate,
                group_idx,
                group_names,
                mask=mask,
                average_corr_over_reps=min_replicates,
            )
        plotter.plot_corr_heatmap(
            "averaged",
            metrics_by_replicate,
            metrics_by_replicate,
            group_idx,
            group_names,
            average_corr_over_reps=min_replicates,
        )

    def plot_cross_job_diff(self):
        diff_plotter = DiffPlots(self.parser, self.key_metric_names)
        diff_plotter.plot_diff_and_variance(
            self.parser.list_jobs()[0], self.parser.list_jobs()[1]
        )
        # job_plots, _, job_metrics, _, _, _ = self.get_cross_job_metrics()
        # for job1_name, job1_metrics, job2_name, job2_metrics in self.cross_job_iter(job_metrics):
        #     # take difference between metrics, plot against job1
        #     diff = {f"{k}-diff": job1_metrics[k] - job2_metrics[k] for k in job1_metrics.keys()}
        #     job_plots.scatter(f"{job1_name}-{job2_name}-id_diff", job1_metrics, diff)
        #     # average abs diff over replicates, plot against carlini metrics
        #     mean_abs_diff = {f"{k}-mabs": np.mean(np.abs(diff[k]), axis=0) for k in diff.keys()}
        #     corr_metrics = {**mean_abs_diff, **self.carlini_metrics}
        #     job_plots.corr_heatmap(f"{job1_name}-{job2_name}-ex_mdiff", corr_metrics, corr_metrics)
        #     job_plots.scatter(f"{job1_name}-{job2_name}-diff_mabs", diff, mean_abs_diff)
        #     job_plots.scatter(f"{job1_name}-{job2_name}-ex_mdiff", self.carlini_metrics, mean_abs_diff)

    def plot_pca(self):
        # load metrics from each job
        for job in self.parser.list_jobs():
            plotter = Plotter(job, subdir=f"pca_plots-ep{job.n_epochs}")
            metrics_dict = Metrics(job, plotter).load_metrics()
            metrics_dict = {n: metrics_dict[n] for n in self.key_metric_names}
            metrics_dict = Metrics.average_over_samples(metrics_dict)
            # average over R, reshape into (N, M)
            n_m_metrics = Metrics.average_over_replicates(metrics_dict)
            n_m_metrics = np.stack(
                [n_m_metrics[n] for n in self.key_metric_names], axis=0
            )
            pca_metrics(plotter, n_m_metrics, self.key_metric_names, transpose=True)
            # reshape into (M*R, N) where R is replicates, M is metrics
            # TODO
            # mr_n_metrics = np.concatenate([metrics_dict[n] for n in metric_names], axis=0)
            # pca_metrics(plotter, mr_n_metrics, metric_names)
