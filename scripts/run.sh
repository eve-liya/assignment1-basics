#!/bin/bash

python ece496b_basics/run_model.py --train_file data/tiny/train.npy \
    --valid_file data/tiny/valid.npy \
    --vocab_size 10000 \
    --d_model 512 --num_heads 16 --num_layers 4 --d_ff 2048 \
    --context_length 256 --batch_size 256 --total_iters 5000 \
    --lr_max 0.001 --lr_min 0 --weight_decay 1e-2\
    --checkpoint_path V100checkpoint.pth --save_interval 1000 \
    --log_interval 100 --eval_interval 500 --eval_iters 100

