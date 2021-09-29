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
from forget.postprocess.train_metrics import PlotTraining


class run_experiment:
    """
    the experiment should call on config.py to get info and create the appropriate directories
    based on the contents of config.ini file. It should then be divided into two steps:
    1. Pretraining (e.g. load model from OpenLTH)
    2. Training (for each job, pass models onto trainer.py which trains it and stores the data)
    """
    def __init__(self):
        parent_dir_path = os.path.dirname(str(os.path.dirname(os.path.realpath(__file__))))
        sys.path.append(os.getcwd()+"/Forget/open_lth/")
        #sys.path.append(str(parent_dir_path) + "/open_lth/")
        print(f"Appending paths: {str(parent_dir_path)}")
        
        #pretraining step:

        #get config files from parser
        self.reader = parser.readConfig()

        #get the number of models to train
        self.num_models = int(self.reader.exp_info["number of models"])
        self.num_jobs = int(self.reader.exp_info["number of jobs"])

        #number of models to train per job
        if self.num_models % self.num_jobs == 0:
            self.num_train_per_job = np.full(self.num_jobs, self.num_models/self.num_jobs).astype(int)
        else:
            self.num_train_per_job = np.full(self.num_jobs - 1, int(self.num_models/self.num_jobs)).astype(int)
            self.num_train_per_job = np.append(self.num_train_per_job, int(self.num_models % self.num_jobs)) #check this

        #make output directories
        self.reader.mk_directories(self.num_train_per_job)

        """
        TRAINING STEP
        """
        print(f"Jobs started at t={datetime.datetime.now()}")
        
        # print(f"Division of jobs (models/job): {self.num_train_per_job}")
        # print(f"Starting training...")
        # #training step:
        # #and for each job, pass models onto trainer
        # for job_idx, job in enumerate(self.reader.jobs):
        #     for model_idx in range(self.num_train_per_job[job_idx]):
        #         print(f"{job}: {self.reader.jobs[job]}, m={model_idx}, t={datetime.datetime.now()}")
        #         model = self.reader.get_model(job)
        #         model_trainer = trainer.train(
        #             model, self.reader.exp_info, self.reader.jobs[job], job_idx, model_idx,
        #             data_dir)
        #         model_trainer.trainLoop(model)
        
        """
        PROCESS STEP
        """
        print(f"Now processing output...")
        # dmg = damagemodel.damageModel()
        # procs = postprocess.postProcess()

        for job in self.reader.list_jobs():
            """New noise evaluation
            """
            # sample_noise(job)
            # eval_noise(job, name_contains=['conv'])

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

            logit_subdirs = ['', 'noise_additive_conv']

            # generate metrics
            for subdir in logit_subdirs:
                metric_generator = PlotTraining(job, noise_dir=subdir)
                metric_generator.gen_and_save_metrics()
            # plot metrics
            metric_generator.plot_metrics(logit_subdirs, 'all')
            metric_generator.plot_metrics(logit_subdirs, 'train')
            metric_generator.plot_metrics(logit_subdirs, 'test')

        print(f"Jobs finished at t={datetime.datetime.now()}")
