#!/bin/bash
#SBATCH --partition=unkillable                      # Ask for unkillable job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=10G
#SBATCH --time=12:00:00
#SBATCH -o %j.out

SRC_DIR=$HOME/proj/forget/Forget

module load python/3.7
module load python/3.7/cuda/10.1/cudnn/8.0/pytorch/1.6.0

virtualenv $SLURM_TMPDIR/env/
source $SLURM_TMPDIR/env/bin/activate

pip install --upgrade pip
pip install -r $SRC_DIR/requirements.txt

python $SRC_DIR/main/run.py
