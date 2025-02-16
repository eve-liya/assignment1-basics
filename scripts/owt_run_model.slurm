#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --partition=kill-shared
## 3 day max run time for community.q, kill.q, exclusive.q, and htc.q. 1 Hour max run time for sb.q
#SBATCH --time=0-1:00:00 ## time format is DD-HH:MM:SS

#SBATCH --cpus-per-task=1
#SBATCH --mem=25G ## max amount of memory per node you require

#SBATCH --gres=gpu:NV-RTX5000:1  ## V100 is hard to get consitently
### To request only 1 of the two GPUs in the node, you would do: gpu:NV-K40:1
## GPU TYPES:
#> sinfo -o %20G
## GRES
## gpu:NV-V100-SXM2:8
## gpu:NV-RTX5000:8
## gpu:NV-L40:2
## gpu:NV-H100:1
## gpu:NV-A30:2
## gpu:NV-RTX-A4000:10
## gpu:NV-RTX2080Ti:2
## gpu:NV-RTX2080Ti:8
## gpu:NV-RTX2080Ti:7
## gpu:NV-RTX2070:8

#SBATCH --output=logs/owt-5000_%j.out   # Output log
#SBATCH --error=logs/owt-5000_%j.err    # Error log

## Useful for remote notification
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=vchan26@hawaii.edu

## All options and environment variables found on schedMD site: http://slurm.schedmd.com/sbatch.html
module purge
module load lang/Anaconda3
source activate eceassignment1

python ece496b_basics/run_model.py --experiment_name OWT --train_file data/owt/train.npy \
    --valid_file data/owt/valid.npy \
    --vocab_size 32000 \
    --d_model 512 --num_heads 16 --num_layers 4 --d_ff 2048 \
    --context_length 256 --batch_size 32 --total_iters 2000 \
    --lr_max 0.001 --weight_decay 1e-2 \
    --checkpoint_path checkpoints/checkpoint-owt-rtx5000.pth --save_interval 100 \
    --log_interval 10 --eval_interval 100 --eval_iters 100

