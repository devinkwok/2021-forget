#!/bin/bash
#SBATCH --partition=unkillable                      # Ask for unkillable job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=10G
#SBATCH --time=30:00:00
#SBATCH --output=forget-%j.out
#SBATCH --error=forget-%j.err

SRC_DIR=$HOME/proj/forget

# load modulels
module load python/3.7
module load python/3.7/cuda/10.1/cudnn/8.0/pytorch/1.6.0

# set up python environment
virtualenv $SLURM_TMPDIR/env/
source $SLURM_TMPDIR/env/bin/activate

# install dependencies
pip install --upgrade pip
pip install -r $SRC_DIR/requirements.txt

# copy training data to node
mkdir $SLURM_TMPDIR/datasets
cp -r $HOME/datasets/cifar10.var/cifar10_torchvision $SLURM_TMPDIR/datasets/

python $SRC_DIR/experiment.py \
    --config_file=$HOME/proj/2021-forget/forget/config/default_config.ini
    --data_dir=$SLURM_TMPDIR/datasets \
