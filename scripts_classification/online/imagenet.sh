#!/bin/bash

for CORRUPTION in 'gaussian_noise' 'shot_noise' 'impulse_noise' 'defocus_blur' 'glass_blur' 'motion_blur' 'zoom_blur' 'snow' 'frost' 'fog' 'brightness' 'contrast' 'elastic_transform' 'pixelate' 'jpeg_compression' 
do
    torchrun --nproc_per_node=2 run_tta.py \
    --source_data_path "../Datasets/imagenet_dataset/" \
    --target_data_path "../Datasets/imagenet_dataset/" \
    --dataset_name imagenet --experiment_dir "../Experiments/Online/ImageNet_C/${CORRUPTION}/5/TeSLA/Resnet50/Seed_0/" \
    --seed 0 \
    --n_epochs 1 \
    --bn_epochs 0 \
    --arch resnet50 \
    --corruption ${CORRUPTION} \
    --corruption_level 5 \
    --port 9966 \
    --lr 0.001 \
    --batch_size 128 \
    --n_classes 1000 \
    --n_neigh 1 \
    --ema_momentum 0.9 \
    --aug_mult_easy 4 \
    --nn_queue_size 2 \
    --sub_policy_dim 2
done
