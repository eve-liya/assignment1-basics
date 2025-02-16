#!/bin/bash
#SBATCH --job-name=330million
#SBATCH --partition=kill-shared
## 3 day max run time for community.q, kill.q, exclusive.q, and htc.q. 1 Hour max run time for sb.q
#SBATCH --time=0-1:00:00 ## time format is DD-HH:MM:SS

#SBATCH --cpus-per-task=1
#SBATCH --mem=25G ## max amount of memory per node you require

#SBATCH --gres=gpu:NV-H100:1  ## V100 is hard to get consitently
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

#SBATCH --output=logs/330_%j.out   # Output log
#SBATCH --error=logs/330_%j.err    # Error log

## Useful for remote notification
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=vchan26@hawaii.edu

## All options and environment variables found on schedMD site: http://slurm.schedmd.com/sbatch.html
module purge
module load lang/Anaconda3
source activate eceassignment1

python ece496b_basics/run_model.py --train_file data/tiny/train.npy \
    --experiment_name H100-exp \
    --valid_file data/tiny/valid.npy \
    --vocab_size 10000 \
    --d_model 512 --num_heads 16 --num_layers 4 --d_ff 2048 \
    --context_length 256 --batch_size 256 --total_iters 5000 \
    --lr_max 0.001 --lr_min 0 --weight_decay 1e-2\
    --checkpoint_path H100checkpoint.pth --save_interval 1000 \
    --log_interval 100 --eval_interval 500 --eval_iters 100
