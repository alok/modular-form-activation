#!/bin/bash
#SBATCH --job-name=eta_nanogpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --partition=h100
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --output=logs/%x-%j.out

module load cuda/12.2   # adjust to cluster modules
module load python/3.12 # or your env setup

# Activate virtual environment
source ~/.venvs/modular-form/bin/activate

# Optional: load W&B API key from secrets
export WANDB_API_KEY=$(cat ~/.config/wandb_api_key)

srun torchrun --standalone --nproc_per_node=8 \
    $(pwd)/scripts/train_nanogpt_eta.py \
    --clone-if-missing \
    -- --batch_size 12 --gradient_accumulation_steps 40 --wandb_log=True 