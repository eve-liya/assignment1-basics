#!/bin/bash

python ece496b_basics/run_model.py --train_file data/tiny/train.npy \
    --valid_file data/tiny/valid.npy \
    --vocab_size 50000 \
    --d_model 512 --num_heads 8 --num_layers 6 --d_ff 2048 \
    --context_length 128 --batch_size 64 --total_iters 10000 \
    --lr_max 0.0005 --lr_min 0.00001 --weight_decay 1e-2\
    --checkpoint_path checkpoint.pth --save_interval 1000 \
    --log_interval 100 --eval_interval 500 --eval_iters 100

