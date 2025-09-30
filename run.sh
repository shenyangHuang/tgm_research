#!/bin/bash
#SBATCH --partition=long #unkillable #main #long
#SBATCH --output=trade_rewire_1.txt 
#SBATCH --error=trade_rewire_1_error.txt 
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --gres=gpu:1                   # Ask for 1 titan xp gpu:rtx8000:1 
#SBATCH --mem=32G #64G                             # Ask for 32 GB of RAM
#SBATCH --time=24:00:00    #48:00:00                   # The job will run for 1 day

module load python/3.10
source $SCRATCH/my_venv/bin/activate
pwd

# python -u examples/linkproppred/rewire_gcn.py --seed=2 --epochs=500 --device=cuda:0 --wandb

# python -u examples/linkproppred/gcn.py --seed=2 --epochs=500 --device=cuda:0 --wandb

python -u examples/nodeproppred/rewire_gcn.py --seed=1 --epochs=500 --device=cuda:0 --wandb
