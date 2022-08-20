import os
import argparse
import datetime

import torch


def build_args():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")

    # device
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--num_workers", default=1, type=int)

    # model
    parser.add_argument('--backbone', type=str, default='xception',
                        choices=['resnet', 'xception'],
                        help='backbone name (default: xception)')
    parser.add_argument('--out_stride', type=int, default=16,
                        help='network output stride (default: 8)')

    # log
    parser.add_argument("--checkpoint", default='1-2-257-100', type=str)
    parser.add_argument('--interval', type=int, default=1,
                        help='saving and evaluation interval (default: 10)')

    # train
    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument('--epoch', default=100, type=int)

    # Optimization
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')

    args = parser.parse_args()
    args.dtype = torch.float32
    if args.checkpoint == '':
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        args.checkpoint = now
    args.checkpoint = os.path.join('./experiment', args.checkpoint)
    os.makedirs(args.checkpoint, exist_ok=True)

    return args
