#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
train_dir="./wrn_28_10_split_1_2_5"
cluster_path="./scripts/clustering_1_2_5.pkl"

python train.py --train_dir $train_dir \
    --batch_size 100 \
    --test_interval 500 \
    --test_iter 100 \
    --num_residual_units 4 \
    --k 10 \
    --cluster_path $cluster_path \
    --l2_weight 0.0005 \
    --initial_lr 0.1 \
    --lr_step_epoch 80.0,120.0 \
    --lr_decay 0.1 \
    --max_steps 100000 \
    --checkpoint_interval 5000 \
    --gpu_fraction 0.96 \
    --display 100 \
