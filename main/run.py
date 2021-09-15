import sys
import os

sys.path.append(os.path.join(os.getcwd(), "Forget", "open_lth"))
sys.path.append(os.getcwd())

from Forget.main import experiment

experiment.run_experiment()