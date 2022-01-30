import os
import datetime
from matplotlib.pyplot import plot
import torch
from forget.damage.example_noise import ExampleIidEraseNoise, ExampleIidGaussianNoise
from forget.main import parser
from forget.postprocess.metrics import Metrics
from forget.postprocess.pair_rep_metrics import PairReplicateMetrics
from forget.training import trainer
from forget.damage.model_noise import ModelNoisePerturbation
from forget.damage.prune import PrunePerturbation
from forget.main.exp_plots import ExperimentPlots
from forget.postprocess.plot_metrics import PlotMetrics
from forget.postprocess.single_rep_metrics import SingleReplicateMetrics


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

            perturbations = [
                ExampleIidEraseNoise(job),
                ExampleIidGaussianNoise(job),
                PrunePerturbation(job),
                ModelNoisePerturbation(job, filter_layer_names_containing=["conv"]),
            ]
            if "do process" in job.hparams:
                """
                PROCESS STEP
                """
                print(f"Now processing output...")

                """Generate and evaluate model with noise perturbations
                """
                for perturbation in perturbations:
                    print("PERTURBATION", perturbation.name)
                    perturbation.gen_logits()

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
            metrics_generator = SingleReplicateMetrics(job, plotter)
            pair_generator = PairReplicateMetrics(job, plotter)
            if "do metrics" in job.hparams:
                """Plot auc, diff, and forgetting ranks"""
                plotter.plot_class_counts(
                    "labels", job.get_eval_labels()
                )  # check label distribution
                metrics_generator.gen_metrics_from_training()
                for perturbation in perturbations:
                    print("PERTURB METRICS", perturbation.name)
                    metrics_generator.gen_metrics_from_perturbation(perturbation)
                metrics_generator.combine_replicates()

                pair_generator.gen_metrics_from_training()
                pair_generator.combine_replicates()

            if "do plot" in job.hparams:
                exp_plots = ExperimentPlots(job, plotter)
                metrics = metrics_generator.load_metrics()
                pair_metrics = pair_generator.load_metrics()
                # summarize noise_metrics over S to get R x N (same as train_metrics)
                averaged_metrics = Metrics.average_over_samples(metrics)
                perturb_metrics = Metrics.filter_metrics(
                    averaged_metrics, "first_forget"
                )
                train_metrics = Metrics.filter_metrics(
                    averaged_metrics, "train-first_learn"
                )
                averaged_key_metrics = {**perturb_metrics, **train_metrics}
                pair_and_single_metrics = {
                    **Metrics.average_over_replicates(averaged_key_metrics),
                    **Metrics.average_over_replicates(
                        Metrics.average_over_samples(pair_metrics)
                    ),
                }
                for cutoff in [1, 30, 70]:
                    if cutoff < job.n_epochs:
                        learned_before_iter = (
                            job.n_epochs - cutoff
                        ) * job.n_iter_per_epoch
                        print(f"Plotting cutoff={cutoff}")
                        exp_plots.plot_learned_before_cutoff(
                            pair_and_single_metrics, learned_before_iter
                        )
                exp_plots.plot_all(averaged_metrics)
                # exp_plots.plot_all(pair_metrics)

                # for perturbation in perturbations:
                #     exp_plots.plot_metrics_by_sample(metrics, perturbation)

        print(f"Jobs finished at t={datetime.datetime.now()}")
