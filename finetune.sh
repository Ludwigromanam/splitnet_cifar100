#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
train_dir="./wrn-16-8_1-2-5"
data_dir="/data1/common_datasets/cifar100/train_val_split/"
cluster_path="./scripts/clustering_1_2_5.pkl"
basemodel_path="./wrn-16-8/model.ckpt-99999"

python finetune.py --train_dir $train_dir \
    --data_dir $data_dir \
    --batch_size 100 \
    --test_interval 500 \
    --test_iter 50 \
    --num_residual_units 2 \
    --k 8 \
    --cluster_path $cluster_path \
    --l2_weight 0.0005 \
    --initial_lr 0.001 \
    --lr_step_epoch 60.0,120.0\
    --lr_decay 0.1 \
    --split_lr_mult 10.0 \
    --max_steps 70000 \
    --checkpoint_interval 5000 \
    --gpu_fraction 0.96 \
    --display 20 \
    --basemodel $basemodel_path
