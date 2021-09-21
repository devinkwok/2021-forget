import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'open_lth'))

from forget.main import experiment

if __name__ == '__main__':
    experiment.run_experiment()
