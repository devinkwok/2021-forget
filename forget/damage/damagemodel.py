import torch
import numpy as np
import os
import sys
from open_lth.foundations import hparams
from open_lth.models import registry
from pathlib import Path
from forget.main import parser


class damageModel:
    def __init__(self, config_file=os.getcwd() + "/Forget/config/default_config.ini"):
        sys.path.append(os.getcwd() + "/Forget/open_lth/")
        print(f"Appending {os.getcwd()}/Forget/open_lth/")
        self.reader = parser.readConfig(config_file)
        self.exp_name = self.reader.exp_info["name"]
        # self.num_jobs = int(self.reader.exp_info["number of jobs"])
        parent_dir_path = Path(Path().absolute()).parent

        self.list_model_folders = []
        print(f"Reading from output files in experiment {self.exp_name}...")
        for job in self.reader.jobs:  # remember to put everything into forgetdata
            job_subdir = [
                str(parent_dir_path) + "/" + self.exp_name + "/" + job + "/" + f.name
                for f in os.scandir(
                    str(parent_dir_path) + "/" + self.exp_name + "/" + job + "/"
                )
                if f.is_dir()
            ]
            for dir in job_subdir:
                self.list_model_folders.append(dir)
            self.max_epoch = int(self.reader.jobs[job]["num epochs"])

            if self.reader.jobs[job]["model parameters"] == "default":
                self.model_hparams = hparams.ModelHparams(
                    "cifar_resnet_20", "kaiming_uniform", "uniform"
                )
            else:
                pass  # to add in

        print(self.list_model_folders)

        for folder in self.list_model_folders:
            self.model = registry.get(self.model_hparams).cuda()
            self.model.load_state_dict(
                torch.load(folder + "/epoch=" + str(self.max_epoch) + ".pt")[
                    "model_state_dict"
                ]
            )
            self.clones = self.addNoise(self.model)

            save_clone_path = folder + "/clones/"
            Path(save_clone_path).mkdir(parents=True, exist_ok=True)
            for idx, clone in enumerate(self.clones):
                torch.save(clone.state_dict(), save_clone_path + str(idx) + ".pt")

            del self.model
            del self.clones

    def addNoise(self, model, num_points=200, min_noise=0.0, max_noise=0.1):
        """
        returns an array of length num_points, consisting of models increasingly damaged
        from Gaussian noise with stdev min_noise to max_noise
        """
        from open_lth.foundations import hparams
        from open_lth.models import registry

        model.eval()
        self.model_clones = []
        model_state_dict = model.state_dict()

        epsilons = np.linspace(min_noise, max_noise, num_points)
        for i in range(len(epsilons)):
            sys.stdout.write("\r{0}Cloning models...".format("|" * i))
            sys.stdout.flush()
            self.model_clones.append(registry.get(self.model_hparams).cuda())
            self.model_clones[i].load_state_dict(model_state_dict)

        with torch.no_grad():
            k = 0
            for model in self.model_clones:
                for param in model.parameters():
                    param.multiply_(
                        1
                        + torch.empty(param.size())
                        .cuda()
                        .normal_(mean=0, std=epsilons[k])
                    )
                k += 1

        for models in self.model_clones:
            models.eval()

        return self.model_clones

    def getEpsilons(self, num_points=200, min_noise=0.0, max_noise=0.1):
        return np.linspace(min_noise, max_noise, num_points)
