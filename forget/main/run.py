import os
import sys
import argparse

sys.path.append(os.getcwd())

from forget.main import experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default="./config/default_config.ini", type=str)
    parser.add_argument('--data_dir', default="./datasets/", type=str)
    args = vars(parser.parse_args())

    experiment.run_experiment(args['config_file'], args['data_dir'])
