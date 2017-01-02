#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
train_dir="./wrn_16_8_split_1_2_5"
data_dir="/data1/common_datasets/cifar100/train_val_split/"
cluster_path="./scripts/clustering_1_2_5.pkl"

python train.py --train_dir $train_dir \
    --data_dir $data_dir \
    --batch_size 100 \
    --test_interval 500 \
    --test_iter 100 \
    --num_residual_units 2 \
    --k 8 \
    --cluster_path $cluster_path \
    --l2_weight 0.0005 \
    --initial_lr 0.1 \
    --lr_step_epoch 80.0,120.0 \
    --lr_decay 0.1 \
    --split_lr_mult 1.0 \
    --max_steps 100000 \
    --checkpoint_interval 5000 \
    --gpu_fraction 0.96 \
    --display 100 \
