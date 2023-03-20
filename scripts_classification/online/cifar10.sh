#!/bin/bash

for CORRUPTION in 'gaussian_noise' 'shot_noise' 'impulse_noise' 'defocus_blur' 'glass_blur' 'motion_blur' 'zoom_blur' 'snow' 'frost' 'fog' 'brightness' 'contrast' 'elastic_transform' 'pixelate' 'jpeg_compression' 
do
    python run_tta.py \
    --target_data_path "../Datasets/cifar_dataset/CIFAR-10-C/" \
    --pretrained_source_path "../Source_classifiers/cifar10/ckpt.pth" \
    --dataset_name cifar10 \
    --experiment_dir "../Experiments/Online/CIFAR10_C/${CORRUPTION}/5/TeSLA/Resnet50/" \
    --seed 0 \
    --batch_size 128 \
    --n_epochs 1 \
    --bn_epochs 0 \
    --arch resnet50_s \
    --corruption ${CORRUPTION} \
    --corruption_level 5 \
    --n_neigh 1 \
    --n_classes 10 \
    --lr 1e-3 \
    --ema_momentum 0.99 \
    --aug_mult_easy 4 \
    --nn_queue_size 2 \
    --sub_policy_dim 2
done
