#!/bin/bash
#SBATCH --partition=unkillable                      # Ask for unkillable job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:rtx8000:1                          # Ask for 1 GPU
#SBATCH --mem=24G
#SBATCH --time=30:00:00
#SBATCH --output=forget-%j.out
#SBATCH --error=forget-%j.err

SRC_DIR=$HOME/proj/2021-forget

# load modulels
module load python/3.7
module load python/3.7/cuda/10.1/cudnn/8.0/pytorch/1.6.0

if ! [ -d "$SLURM_TMPDIR/env/" ]; then
    # set up python environment
    virtualenv $SLURM_TMPDIR/env/
    source $SLURM_TMPDIR/env/bin/activate
    # install dependencies
    pip install --upgrade pip
    pip install -r $SRC_DIR/requirements.txt
else
    source $SLURM_TMPDIR/env/bin/activate
fi

if ! [ -d "$SLURM_TMPDIR/data" ]; then
    # copy training data to node
    mkdir $SLURM_TMPDIR/data
    cp -r $SRC_DIR/data $SLURM_TMPDIR/
fi

python $SRC_DIR/forget/main/run.py \
    --config_file=$SRC_DIR/config/default_config.ini \
    --data_dir=$SLURM_TMPDIR/data \
    --out_dir=$HOME/scratch \
