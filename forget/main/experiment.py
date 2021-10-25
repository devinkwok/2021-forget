from io import IncrementalNewlineDecoder
import os
import sys
import datetime
import numpy as np
from forget.main import parser
from forget.training import trainer
from forget.damage import damagemodel
from forget.postprocess import postprocess
from forget.damage.noise import sample_noise, eval_noise
from forget.postprocess.weight_stats import PlotWeights
from forget.postprocess.train_metrics import GenerateMetrics


class run_experiment:
    """
    the experiment should call on config.py to get info and create the appropriate directories
    based on the contents of config.ini file. It should then be divided into two steps:
    1. Pretraining (e.g. load model from OpenLTH)
    2. Training (for each job, pass models onto trainer.py which trains it and stores the data)
    """
    def __init__(self):
        #get config files from parser
        self.parser = parser.readConfig()

        print(f"Jobs started at t={datetime.datetime.now()}")
        for job in self.parser.list_jobs():

            """
            TRAINING STEP
            """
            print(f"Starting training...")
            for model_idx in range(job.n_replicates):
                print(f"{job.name} m={model_idx}, t={datetime.datetime.now()}")
                model_trainer = trainer.train(job.get_model(), job, model_idx)
                model_trainer.trainLoop()

            """
            PROCESS STEP
            """
            print(f"Now processing output...")

            """New noise evaluation
            """
            sample_noise(job)
            eval_noise(job, name_contains=['conv'])

            """Plot model weights
            """
            # plot_weights = PlotWeights(job)
            # plot_weights.plot_all(
            #     ['conv', 'fc.weight', 'shortcut.0.weight'],
            #     ['bn1.weight', 'bn2.weight'],
            #     ['bn1.bias', 'bn2.bias'],
            #     )
            # # noisy model weights
            # plot_weights = PlotWeights(job, noise_subdir='noise_additive_conv')
            # plot_weights.plot_all(
            #     ['conv', 'fc.weight', 'shortcut.0.weight'],
            #     ['bn1.weight', 'bn2.weight'],
            #     ['bn1.bias', 'bn2.bias'],
            #     )

            """Plot auc, diff, and forgetting ranks
            """
            # gen_metrics = GenerateMetrics(job, force_generate=False)
            # train_metrics, metrics_by_epoch = gen_metrics.gen_train_metrics_by_epoch()
            # noise_metrics = gen_metrics.gen_noise_metrics(
            #                 job.hparams['noise type'], ['conv'])
            # gen_metrics.gen_train_to_noise_metrics(train_metrics, metrics_by_epoch, noise_metrics)
            # noise_metrics = gen_metrics.gen_noise_metrics(
            #                 job.hparams['noise type'], [])

        print(f"Jobs finished at t={datetime.datetime.now()}")
