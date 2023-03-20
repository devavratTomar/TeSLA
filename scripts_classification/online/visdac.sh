#!/bin/bash
for SEED in 43 44 45
do
    torchrun --nproc_per_node=2 run_tta.py \
    --target_data_path "../Datasets/visda_dataset" \
    --pretrained_source_path "../Source_classifiers/VisDA/best_seed_${SEED}.pth" \
    --dataset_name visda \
    --experiment_dir "../Experiments/Online/VisDA/SLAug/Resnet101/Seed_${SEED}" \
    --seed ${SEED} \
    --batch_size 128 \
    --n_epochs 1 \
    --arch resnet101 \
    --port 30500 \
    --n_classes 12 \
    --ema_momentum 0.9 \
    --lr 1e-3 \
    --num_workers 8 \
    --bn_epochs 0 \
    --n_neigh 10 \
    --aug_mult_easy 4 \
    --sub_policy_dim 4 \
    --nn_queue_size 256
done