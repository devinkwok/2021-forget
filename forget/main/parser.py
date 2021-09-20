from dataclasses import dataclass
from pathlib import Path
import configparser
import os
import sys

@dataclass
class readConfig:
    config_file: str = os.getcwd()+"/Forget/config/default_config.ini"
    
    def __post_init__(self):
        print(f"Current working directory: {os.getcwd()}")
        sys.path.append(str(os.path.dirname(os.path.realpath(__file__))))
        parent_dir_path = os.path.dirname(os.getcwd())
        #parent_dir_path = Path(Path().absolute()).parent
        #parent_dir_path = os.path.dirname(os.getcwd())
        #sys.path.append(str(parent_dir_path) + '/open_lth/') #change this

        config = configparser.ConfigParser()
        config.read(self.config_file)
        self.sections = config.sections()
        self.exp_info = {}
        self.jobs = {}


        for section in self.sections:
            if section == "Experiment info":
                options = config.options(section)
                self.exp_info[options[0]] = config.get(section, options[0])
                self.exp_info[options[1]] = config.get(section, options[1])
                self.exp_info[options[2]] = int(config.get(section, options[2]))
                self.exp_info[options[3]] = int(config.get(section, options[3]))
            elif str.split(section)[0] == "Job" and str.split(section)[1].isdigit():
                self.jobs[section] = {}
                options = config.options(section)
                for i in range(len(options)):
                    self.jobs[section][str(options[i])] = config.get(section, options[i])
            else:
                raise ValueError("Unknown section command in config file!")
    
        #make directories for experiment, each job and model
        parent_dir_path = Path(Path().absolute()).parent

        #make experiment path
        self.exp_path = str(parent_dir_path) + "/" + self.exp_info["name"]
        print(f"Experiment info and path: {self.exp_info} , {self.exp_path}")
        Path(self.exp_path).mkdir(parents=True, exist_ok=True)

        #for each job and for each model in the job, make the corresponding directory
    
    def mk_directories(self, model_numbers_per_job):
        for job in self.jobs:
            self.job_path = self.exp_path + "/" + job
            Path(self.job_path).mkdir(parents=True, exist_ok=True)
            for numbers_per_job in model_numbers_per_job:
                for model_idx in range(numbers_per_job):#range(int(self.exp_info["number of models"])):
                    self.model_path = self.job_path + "/model" + str(model_idx)
                    Path(self.model_path).mkdir(parents=True, exist_ok=True)
                #and if track flags are on, create directories for those
                    if self.jobs[job]["measure forget"] == "true" or self.jobs[job]["measure forget"] == "True":
                        Path(self.model_path + "/forgetdata").mkdir(parents=True, exist_ok=True)
                    if self.jobs[job]["track correct examples"] == "true" or self.jobs[job]["track correct examples"] == "True":
                        Path(self.model_path + "/correctdata").mkdir(parents=True, exist_ok=True)
        
        #model params
    def get_model(self, job):
        from open_lth.foundations import hparams
        from open_lth.models import registry

        if self.jobs[job]["model parameters"] == "default":
            _model_params = hparams.ModelHparams(
                'cifar_resnet_20',
                'kaiming_uniform',
                'uniform'
            )
            return registry.get(_model_params).cuda()
        else:
            pass #to do for case of other model parameters
