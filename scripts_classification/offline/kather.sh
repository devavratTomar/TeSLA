#!/bin/bash

for SEED in 0 1 2 3 4 5 6 7 8 9
do
    python run_tta.py \
    --target_data_path "../Datasets/Kather/kather2016" \
    --pretrained_source_path "../Source_classifiers/Kather/seed_${SEED}/checkpoint/net_state_dict_ep20_acc0.00.pth" \
    --dataset_name kather \
    --experiment_dir "../Experiments/Offline/Kather/TeSLA/MobileNetV2/Seed_${SEED}/" \
    --seed ${SEED} \
    --n_epochs 70 \
    --bn_epochs 2 \
    --arch mobilenet_v2 \
    --lr 0.005 \
    --batch_size 32 \
    --n_classes 4 \
    --num_workers 0 \
    --ema_momentum 0.96 \
    --n_neigh 8 \
    --nn_queue_size 256 \
    --sub_policy_dim 2 \
    --apply_lr_scheduler
done
