
import argparse

from test_time_trainers.tesla_seg import TeSLA_Seg
from utilities.utils import setup_experiment


def get_opt_parser():
    
    parser = argparse.ArgumentParser("TeSLA for Segmentation", add_help=False)
    parser.add_argument("--target_data_path", type=str, help="Root Directory of the Dataset")
    parser.add_argument("--target_info_path", type=str, default="")
    parser.add_argument("--target_list_path", type=str, default="")
    parser.add_argument("--target_eval_list_path", type=str, default="")
    parser.add_argument("--target_input_size", default=[1024//2, 512//2])
    
    parser.add_argument("--pretrained_source_path", type=str, help="Path to the trained source model")
    parser.add_argument("--experiment_dir", type=str, help="Path to the experiment directory")
    
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_type", type=str, choices=["med", "rgb"], default="rgb")
    parser.add_argument("--target_sites")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_neigh", type=int, default=1)
    parser.add_argument("--ema_momentum", type=float, default=0.999)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_channels", type=int, default=3)
    parser.add_argument("--n_classes", type=int)
    parser.add_argument("--sub_policy_dim", type=int, default=3)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--aug_mult", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--apply_lr_scheduler", action="store_true", help="Whether ot not to apply learning rate scheduler")
    parser.add_argument("--nn_queue_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--lambda_1", type=float, default=1.0)
    parser.add_argument("--lambda_2", type=float, default=1.0)

    # Arguments for distributed training
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--dist_url", type=str, default="env://")
    
    parser.add_argument("--debug", action="store_true", help="Debug Augmented Images")
    parser.add_argument("--replay_policy", action="store_true", help="Replay Past Augmentation Policies")
    parser.add_argument("--weak_mult", type=int, default=3)
    

    return parser


def test_time_adapt(opt):
    setup_experiment(opt)
    trainer = TeSLA_Seg(opt)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('TeSLA', parents=[get_opt_parser()])
    opt = parser.parse_args()

    if opt.dataset_type == "med":
        if opt.dataset_name == 'prostate':
            opt.target_sites = ['site-'+ site_nbr for site_nbr in opt.target_sites.split(',')]
        else:
            opt.target_sites = ['site'+ site_nbr for site_nbr in opt.target_sites.split(',')]
        
    test_time_adapt(opt)
    
