#!/bin/bash
#SBATCH --partition=unkillable                      # Ask for unkillable job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=24G
#SBATCH --time=30:00:00
#SBATCH --output=forget-%j.out
#SBATCH --error=forget-%j.err

SRC_DIR=$HOME/proj/2021-forget

# load modulels
module load python/3.7
module load python/3.7/cuda/10.1/cudnn/8.0/pytorch/1.6.0

# set up python environment
virtualenv $SLURM_TMPDIR/env/
source $SLURM_TMPDIR/env/bin/activate

# install dependencies
pip install --upgrade pip
pip install -r $SRC_DIR/requirements.txt

#FIXME $SLURM_TMPDIR doesn't work with torchvision
# copy training data to node
# mkdir $SLURM_TMPDIR/data
# cp -r $HOME/datasets/cifar10.var/cifar10_torchvision/cifar-10-batches-py $SLURM_TMPDIR/data/

python $SRC_DIR/forget/main/run.py \
    --config_file=$SRC_DIR/config/exp_noise_perturb.ini \
    --data_dir=$SRC_DIR/data \
    --out_dir=$HOME/out \
