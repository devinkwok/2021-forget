import datetime
from forget.metrics.eval_metrics import EvalMetrics

from forget.perturb.example_noise import ExampleIidEraseNoise, ExampleIidGaussianNoise
from forget.main import parser
from forget.metrics.tvd_metrics import TotalVariationDistanceMetrics
from forget.training import trainer
from forget.perturb.model_noise import ModelNoisePerturbation
from forget.perturb.prune import PrunePerturbation
from forget.plot.plotter import Plotter
from forget.metrics.train_metrics import TrainMetrics
from forget.metrics.perturb_metrics import PerturbMetrics
from forget.training.eval import plot_training, evaluate_trained_model


class run_experiment:
    """
    the experiment should call on config.py to get info and create the appropriate directories
    based on the contents of config.ini file. It should then be divided into two steps:
    1. Pretraining (e.g. load model from OpenLTH)
    2. Training (for each job, pass models onto trainer.py which trains it and stores the data)
    """

    @staticmethod
    def perturbations(job):
        return [
            ExampleIidEraseNoise(job),
            ExampleIidGaussianNoise(job),
            PrunePerturbation(job),
            ModelNoisePerturbation(job, filter_layer_names_containing=["conv"]),
        ]

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
                if job.hparams["model init"] == "fixed":

                    def get_fixed_init():
                        return job.get_model().state_dict()

                    fixed_init = job.cached(
                        get_fixed_init, "fixed_init", f"model_state_dict.pt"
                    )
                else:
                    fixed_init = None
                model_trainer = trainer.train(job)
                for _, replicate_dir in job.replicates():
                    if not model_trainer.is_finished(replicate_dir):
                        print(
                            f"{job.name} {replicate_dir}, t={datetime.datetime.now()}"
                        )
                        model_trainer.trainLoop(replicate_dir, fixed_init=fixed_init)
                evaluate_trained_model(job)
                plot_training(job)

            perturbations = self.perturbations(job)
            if "do perturb" in job.hparams:
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

            plotter = Plotter(job, subdir=f"plots-ep{job.n_epochs}")
            train_metrics = TrainMetrics(job, plotter)
            perturb_metrics = PerturbMetrics(job, plotter)
            eval_metrics = EvalMetrics(job, plotter)
            tvd_metrics = TotalVariationDistanceMetrics(job, plotter)
            if "do metrics" in job.hparams:
                """Plot auc, diff, and forgetting ranks"""
                plotter.plot_class_counts(
                    "labels", job.get_eval_labels()
                )  # check label distribution
                train_metrics.gen_metrics()
                for perturbation in perturbations:
                    print("PERTURB METRICS", perturbation.name)
                    perturb_metrics.gen_metrics(perturbation)
                eval_metrics.gen_metrics()
                tvd_metrics.gen_metrics()

                train_metrics.combine_replicates()

        print(f"Experiment finished at t={datetime.datetime.now()}")
