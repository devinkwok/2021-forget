import os
import datetime
import torch
from forget.main import parser
from forget.training import trainer
from forget.damage.noise import NoisePerturbation
from forget.damage.prune import PrunePerturbation
from forget.main import noise_plots
from forget.postprocess.plot_metrics import PlotMetrics
from forget.postprocess.train_metrics import GenerateMetrics


class run_experiment:
    """
    the experiment should call on config.py to get info and create the appropriate directories
    based on the contents of config.ini file. It should then be divided into two steps:
    1. Pretraining (e.g. load model from OpenLTH)
    2. Training (for each job, pass models onto trainer.py which trains it and stores the data)
    """

    def __init__(self):
        # get config files from parser
        self.parser = parser.readConfig()

        print(f"Jobs started at t={datetime.datetime.now()}")
        for job in self.parser.list_jobs():
            if "do train" in job.hparams:
                """
                TRAINING STEP
                """
                print(f"Starting training...")
                for model_idx in range(job.n_replicates):
                    print(f"{job.name} m={model_idx}, t={datetime.datetime.now()}")
                    model_trainer = trainer.train(job, model_idx)
                    model_trainer.trainLoop()

            noise_exp = NoisePerturbation(job, filter_layer_names_containing=["conv"])
            prune_exp = PrunePerturbation(job)
            if "do process" in job.hparams:
                """
                PROCESS STEP
                """
                print(f"Now processing output...")

                """Generate and evaluate model with noise perturbations
                """
                noise_exp.gen_noise_logits()
                prune_exp.gen_prune_logits()

                """Plot model weights
                """
                # plot_weights = PlotWeights(job)
                # plot_weights.plot_all(
                #     ['conv', 'fc.weight', 'shortcut.0.weight'],
                #     ['bn1.weight', 'bn2.weight'],
                #     ['bn1.bias', 'bn2.bias'],
                #     )

                # plot_weights = PlotWeights(job, noise_subdir='noise_additive_conv')
                # plot_weights.plot_all(
                #     ['conv', 'fc.weight', 'shortcut.0.weight'],
                #     ['bn1.weight', 'bn2.weight'],
                #     ['bn1.bias', 'bn2.bias'],
                #     )

            plotter = PlotMetrics(job, subdir=f"plots-ep{job.n_epochs}")
            if "do metrics" in job.hparams:
                """Plot auc, diff, and forgetting ranks"""
                gen = GenerateMetrics(job, plotter)
                plotter.plot_class_counts(
                    "labels", gen.labels
                )  # check label distribution

                gen.gen_train_metrics()
                gen.gen_noise_metrics(noise_exp.subdir, noise_exp.scales)
                gen.gen_prune_metrics(prune_exp.subdir, prune_exp.scales)

            if "do plot" in job.hparams:
                metrics_dir = os.path.join(job.save_path, f"metrics-ep{job.n_epochs}")
                files = os.listdir(metrics_dir)
                metrics = {
                    f[: -len(".metric")]: torch.load(
                        os.path.join(metrics_dir, f), map_location=torch.device("cpu")
                    )
                    for f in files
                    if f.endswith(".metric")
                }
                noise_metric_names = filter(
                    lambda x: x.startswith("noise"), metrics.keys()
                )
                noise_metrics = {n: metrics[n] for n in noise_metric_names}
                # summarize noise_metrics over S to get R x N (same as train_metrics)
                # noise_plots.plot_noise_metrics_by_sample(plotter, noise_metrics)
                noise_metrics = noise_plots.average_noise_metrics(noise_metrics)
                for k, v in noise_metrics.items():
                    metrics[k] = v
                for k, v in metrics.items():
                    if len(v.shape) != 2:
                        metrics[k] = v.squeeze()
                        assert len(metrics[k].shape) == 2
                    print(k, metrics[k].shape)

                for cutoff in [
                    0,
                    job.n_iter_per_epoch,
                    10 * job.n_iter_per_epoch,
                    30 * job.n_iter_per_epoch,
                    70 * job.n_iter_per_epoch,
                ]:
                    learned_before_iter = job.n_epochs * job.n_iter_per_epoch - cutoff
                    print(
                        "Plotting with learned cutoff at",
                        cutoff,
                        "iterations",
                        "it",
                        learned_before_iter,
                    )
                    noise_plots.plots_2021_11_24(plotter, metrics, learned_before_iter)
                noise_plots.plot_comparisons(plotter, metrics)
                plotter.plot_metric_rank_qq(metrics)

        print(f"Jobs finished at t={datetime.datetime.now()}")
