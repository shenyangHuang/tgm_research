#!/bin/bash
#SBATCH --partition=unkillable #unkillable #main #long
#SBATCH --output=wiki_rewire_1.txt 
#SBATCH --error=wiki_rewire_1_error.txt 
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --gres=gpu:1                   # Ask for 1 titan xp gpu:rtx8000:1 
#SBATCH --mem=32G #64G                             # Ask for 32 GB of RAM
#SBATCH --time=24:00:00    #48:00:00                   # The job will run for 1 day

module load python/3.10
source $SCRATCH/my_venv/bin/activate
pwd

python -u gcn_rewired.py --seed=1 --epochs=100 --device=cuda:0 --wandb