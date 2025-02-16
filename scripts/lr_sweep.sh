#!/bin/bash

# List of learning rates to sweep over
LEARNING_RATES=(0.0001 0.0002 0.0005 0.001 0.002 0.005)

# Loop over each learning rate and submit a job
for LR in "${LEARNING_RATES[@]}"; do
    sbatch --export=LR_MAX=$LR gpu.slurm
done
