#!/bin/bash

# SPINAL CORD EXPERIMENT

for TARGET in 2 3 4
do
    torchrun --nproc_per_node=2 run_tta_seg.py \
    --target_data_path ../Datasets/MRI/SpinalCord/ \
    --pretrained_source_path ../Source_Segmentation/MRI/SpinalCord/Site1/Segmentor_250000.pth \
    --experiment_dir ../Experiments_Segmentation/Online/SpinalCord/Site_1_to_${TARGET}/ \
    --dataset_name spinalcord \
    --dataset_type med \
    --target_sites ${TARGET} \
    --ema_momentum 0.996 \
    --n_channels 1 \
    --n_classes 3 \
    --n_epochs 1 \
    --lr 0.0002 \
    --batch_size 16 \
    --bn_epochs 0 \
    --weak_mult 5
done

for TARGET in UCL BIDMC HK
do
    torchrun --nproc_per_node=2 run_tta_seg.py \
    --target_data_path ../Datasets/MRI/Prostate \
    --pretrained_source_path ../Source_Segmentation/MRI/Prostate/Site_ISBI/Segmentor_250000.pth \
    --experiment_dir ../Experiments_Segmentation/Online/Prostate/Site_ISBI_to_${TARGET}/ \
    --dataset_name prostate \
    --dataset_type med \
    --target_sites ${TARGET} \
    --ema_momentum 0.996 \
    --n_channels 1 \
    --n_classes 3 \
    --n_epochs 1 \
    --lr 0.0002 \
    --batch_size 16 \
    --bn_epochs 0 \
    --weak_mult 5
done