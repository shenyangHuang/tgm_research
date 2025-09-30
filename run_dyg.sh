#!/bin/bash
#SBATCH --partition=long #unkillable #main #long
#SBATCH --output=dygformer_coin_3.txt 
#SBATCH --error=dygformer_coin_3_error.txt 
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --gres=gpu:a100l:1                   # Ask for 1 titan xp gpu:rtx8000:1 
#SBATCH --mem=32G #64G                             # Ask for 32 GB of RAM
#SBATCH --time=24:00:00    #48:00:00                   # The job will run for 1 day

module load python/3.10
source $SCRATCH/my_venv/bin/activate
pwd

python realtg_dygformer.py --seed=3 --epochs=30 --device=cuda:0 --dataset=tgbl-coin
