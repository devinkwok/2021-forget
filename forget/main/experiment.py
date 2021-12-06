import datetime
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

            """
            TRAINING STEP
            """
            print(f"Starting training...")
            for model_idx in range(job.n_replicates):
                print(f"{job.name} m={model_idx}, t={datetime.datetime.now()}")
                model_trainer = trainer.train(job, model_idx)
                model_trainer.trainLoop()

            """
            PROCESS STEP
            """
            print(f"Now processing output...")

            """Generate and evaluate model with noise perturbations
            """
            noise_exp = NoisePerturbation(job, filter_layer_names_containing=["conv"])
            noise_exp.noise_logits()
            prune_exp = PrunePerturbation(job)
            prune_exp.prune_logits()

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

            """Plot auc, diff, and forgetting ranks
            """
            gen = GenerateMetrics(job)
            plotter = PlotMetrics(job, subdir="plot-metrics-noise")
            plotter.plot_class_counts("labels", gen.labels)  # check label distribution

            train_metrics, n_epoch, n_iter_per_epoch = gen.gen_train_metrics()
            noise_metrics = gen.gen_noise_metrics(noise_exp.subdir, noise_exp.scales)
            prune_metrics = gen.gen_prune_metrics(prune_exp.subdir, prune_exp.scales)
            # summarize noise_metrics over S to get R x N (same as train_metrics)
            noise_metrics = noise_plots.plot_noise_metrics_by_sample(
                plotter, noise_metrics
            )
            metrics = {**train_metrics, **noise_metrics, **prune_metrics}
            for cutoff in [0, 1, n_iter_per_epoch, 2*n_iter_per_epoch,
                        5*n_iter_per_epoch, 10*n_iter_per_epoch]:
                for always_learned in [True, False]:
                    learned_before_iter = n_epoch * n_iter_per_epoch - cutoff
                    noise_plots.plots_2021_11_24(plotter, metrics,
                            learned_before_iter, always_learned=always_learned)
            noise_plots.plot_comparisons(plotter, metrics)
            plotter.plot_metric_rank_qq(metrics)

        print(f"Jobs finished at t={datetime.datetime.now()}")
