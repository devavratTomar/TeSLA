#!/bin/bash

for SEED in 42 43 44
do
    torchrun --nproc_per_node=2 run_tta_seg.py \
    --target_data_path ../Datasets/visda_segmentation_dataset/Cityscapes/ \
    --target_info_path ./dataloaders/segmentation/visda/cityscapes_list/info.json \
    --target_list_path ./dataloaders/segmentation/visda/cityscapes_list/train.txt \
    --target_eval_list_path ./dataloaders/segmentation/visda/cityscapes_list/val.txt \
    --pretrained_source_path ../Source_Segmentation/VisDA/seed_${SEED}/deeplab_epoch_5_lr_0.0002 \
    --experiment_dir ../Experiments_Segmentation/Offline/visdas/seed_${SEED} \
    --dataset_name vidas \
    --dataset_type rgb \
    --target_sites cityscape \
    --n_classes 19 \
    --seed ${SEED} \
    --sub_policy_dim 3 \
    --ema_momentum 0.999 \
    --n_epochs 3 \
    --bn_epochs 2 \
    --weak_mult 3
done