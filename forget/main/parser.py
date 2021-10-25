import os
import sys
import configparser
from pathlib import Path
import argparse
from forget.job import Job


class readConfig:

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config_file', default="./config/default_config.ini", type=str)
        parser.add_argument('--data_dir', default="./datasets/", type=str)
        parser.add_argument('--out_dir', default="./outputs/", type=str)
        self.args = vars(parser.parse_args())
        print(f'Runtime args: {self.args}')
        self._parse_configs()
        self._make_paths()

    def _parse_configs(self):
        config = configparser.ConfigParser()
        config.read(self.args['config_file'])
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

    def _make_paths(self):
        self.exp_path = os.path.join(self.args['out_dir'], self.exp_info["name"])
        print(f"Experiment info and path: {self.exp_info} , {self.exp_path}")
        Path(self.exp_path).mkdir(parents=True, exist_ok=True)
        #for each job and for each model in the job, make the corresponding directory
        for job in self.jobs:
            job_path = os.path.join(self.exp_path, job)
            Path(job_path).mkdir(parents=True, exist_ok=True)

    def list_jobs(self):
        job_list = []
        for job_name in self.jobs:
            hparams = self.jobs[job_name]
            job_list.append(
                Job(job_name, self.exp_path, self.args['data_dir'], hparams))
        return job_list
