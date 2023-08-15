"""Configurations."""

import torch
import argparse
from pathlib import Path


def build_parser():
    """Get arguments from cmd."""
    parser = argparse.ArgumentParser()

    # Arguments related to Data.
    parser.add_argument('--data',
                        type=str,
                        default='cifar10',
                        choices=['cifar10', 'cifar100', 'svhn', 'stl10'],
                        help="Dataset for experiments")
    parser.add_argument('--num_classes',
                        type=int,
                        default=None,
                        help="Number of classes")
    parser.add_argument('--num_X',
                        type=int,
                        default=250,
                        help="Number of labeled dataset")
    parser.add_argument('--include_x_in_u',
                        default=False,
                        action='store_true',
                        help="Inlucde labeled data(X) in unlabeled data(U).")
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help="Batch size of X")
    parser.add_argument('--mu',
                        type=float,
                        default=7,
                        help="Relative size of U")
    parser.add_argument('--augs',
                        type=int,
                        nargs='+',
                        default=[1, 2],
                        help="augmentations (weak: 1, strong: 2)")

    # Arguments related to Network.
    parser.add_argument('--network',
                        type=str,
                        default='wrn_28_2',
                        choices=['squeezenet',
                                 'efficientnet',
                                 'convnext',
                                 'mobilenet',
                                 'shufflenet',
                                 'wrn_28_2',
                                 'wrn_28_8'],
                        help="Network architecture")
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help="Exponential moving average of model weights")

    # Arguments related to Optimization.
    parser.add_argument('--lr',
                        type=float,
                        default=0.03,
                        help="learning rate")
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help="momentum")
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0005,
                        help="weight decay")
    parser.add_argument('--nesterov',
                        default=False,
                        action='store_true',
                        help="nesterov")
    parser.add_argument('--iterations',
                        type=int,
                        default=2**20,
                        help="Number of training iterations.")

    # Arguments related to FlexMatch Algorithm.
    parser.add_argument('--threshold',
                        type=float,
                        default=0.95,
                        help="threshold to generate artificial label")
    parser.add_argument('--lu_weight',
                        type=float,
                        default=1.0,
                        help="unsupervised loss weight")
    parser.add_argument('--mapping',
                        type=str,
                        default='convex',
                        choices=['convex', 'concave', 'linear'],
                        help="Beta Mappging function.")


    # Arguments related to Misc.
    parser.add_argument('--save_path',
                        type=Path,
                        default='./model-store',
                        help="model save path")
    parser.add_argument('--load_path',
                        type=Path,
                        help="model load path for 'resume' or 'eval'")
    parser.add_argument('--print_interval',
                        type=int,
                        default=1000,
                        help="Print log step")
    parser.add_argument('--amp',
                        default=False,
                        action='store_true',
                        help="amp usage flag")
    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=['train', 'eval', 'resume'],
                        help="Runtime mode")

    # Arguments related to wandb.
    parser.add_argument('--wandb',
                        default=False,
                        action='store_true',
                        help="wandb usage flag")
    parser.add_argument('--wb_project',
                        type=str,
                        default='FlexMatch',
                        help="Project Name")
    parser.add_argument('--wb_tags',
                        type=str,
                        nargs='+',
                        default=None,
                        help="Tags of this run")

    return parser.parse_args()


def get_parameters():
    """Get parameters to run."""
    args = build_parser()

    args.save_path.mkdir(exist_ok=True)

    if args.mode == 'resume':
        mode = args.mode
        load_path = args.load_path
        ckpt = torch.load(load_path, map_location='cpu')
        start_iter = ckpt['iteration']
        args = ckpt['args']
        args.mode = mode
        args.load_path = load_path
        args.iterations = max(args.iterations - start_iter, 0)
        del ckpt

    elif args.mode == 'eval':
        mode = args.mode
        load_path = args.load_path
        ckpt = torch.load(args.load_path, map_location='cpu')
        args = ckpt['args']
        args.mode = mode
        args.load_path = load_path
        del ckpt

    # dependent parameters
    args.num_classes = 100 if args.data == 'cifar100' else 10

    # print
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    return args
