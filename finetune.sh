#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
train_dir="/data1/solrefa/tensorflow_models/splitnet/wrn_28_12_split_1_2_5"
cluster_path="./scripts/clustering_1_2_5.pkl"

python finetune.py --train_dir $train_dir \
    --batch_size 100 \
    --test_interval 500 \
    --test_iter 100 \
    --num_residual_units 4 \
    --k 12 \
    --cluster_path $cluster_path \
    --l2_weight 0.0005 \
    --initial_lr 0.001 \
    --lr_step_epoch 10000\
    --lr_decay 0.1 \
    --max_steps 100000 \
    --checkpoint_interval 5000 \
    --gpu_fraction 0.96 \
    --display 20 \
    --basemodel /data1/solrefa/tensorflow_models/splitnet/wrn_28_12_split_1_2_5/wrn_28_12.ckpt-99999
