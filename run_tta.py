
import torch
import argparse
import os

from test_time_trainers.tesla import TeSLA
from utilities.utils import setup_experiment


def get_opt_parser():
    
    parser = argparse.ArgumentParser("TeSLA", add_help=False)

    # Experiment variables
    parser.add_argument("--source_data_path", type=str, help="Root Directory of the Source Dataset")
    parser.add_argument("--target_data_path", type=str, help="Root Directory of the Dataset")
    parser.add_argument("--pretrained_source_path", type=str, help="Path to the trained source model")
    parser.add_argument("--experiment_dir", type=str, help="Path to the experiment directory")
    parser.add_argument("--save_every", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="Debug Augmented Images")    

    # Dataset variables
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--corruption", type=str, default="snow")
    parser.add_argument("--corruption_level", type=int, default=5)

    # Optimization parameters
    parser.add_argument("--arch", type=str, default="resnet50_s")
    parser.add_argument("--n_channels", type=int, default=3)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--bn_epochs", type=int, default=0, help="Number of epochs to calibrate BN layers")
    parser.add_argument("--stop_epochs", type=int, default=25, help="Number of epochs after which we can stop the optimization")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--apply_lr_scheduler", action="store_true", help="Whether ot not to apply learning rate scheduler")

    # Hyper-parameters
    parser.add_argument("--n_neigh", type=int, default=10)
    parser.add_argument("--ema_momentum", type=float, default=0.999)
    parser.add_argument("--sub_policy_dim", type=int, default=2)
    parser.add_argument("--aug_mult_easy", type=int, default=4)
    parser.add_argument("--nn_queue_size", type=int, default=256)
    parser.add_argument("--lmb_norm", type=float, default=1)
    parser.add_argument("--lmb_kl", type=float, default=1)

    # Ablation
    parser.add_argument("--pl_ce", action="store_true", help="PL with cross entropy")
    parser.add_argument("--pl_fce", action="store_true", help="PL with cross entropy")
    parser.add_argument("--no_kl_hard", action="store_true", help="do not use kl divergence on optimal hard augmentation")
    parser.add_argument("--hard_augment", type=str, default="optimal", help="type of hard augmentation to use", choices=("optimal", "aa", "randaugment"))
    parser.add_argument("--sanity_check", action="store_true")

    # Source Stats Hyper-parameters
    parser.add_argument("--use_source_stats", action="store_true")
    parser.add_argument("--scale_ext", type=float, default=0.05)

    # Distributed training parameters
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--dist_url", type=str, default="env://")
    parser.add_argument("--port", type=str, default="29500")

    return parser


def test_time_adapt(opt):

    setup_experiment(opt)
    trainer = TeSLA(opt)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('TeSLA-Offline', parents=[get_opt_parser()])
    opt = parser.parse_args()
    test_time_adapt(opt)

