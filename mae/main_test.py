# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Test script for MAE fine-tuned models
# --------------------------------------------------------

import argparse
import datetime
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import util.misc as misc
from util.datasets import build_test_dataset
from util.pos_embed import interpolate_pos_embed

import models_vit

from engine_finetune import evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAE test script for image classification', add_help=False)
    
    # Model parameters
    parser.add_argument('--model', default='vit_base_patch4', type=str, metavar='MODEL',
                        help='Name of model to test')
    parser.add_argument('--input_size', default=64, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    # Dataset parameters
    parser.add_argument('--data_path', default='/data/lhs1208/mae_res_64/data', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=200, type=int,
                        help='number of the classification types')
    
    # Checkpoint parameters
    parser.add_argument('--resume', default='', required=True,
                        help='resume from checkpoint (required)')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--device', default='cuda',
                        help='device to use for testing')
    parser.add_argument('--seed', default=0, type=int)
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    
    return parser


def main(args):
    misc.init_distributed_mode(args)
    
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)
    
    cudnn.benchmark = True
    
    # Build test dataset
    dataset_test = build_test_dataset(args)
    
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if args.dist_eval:
            if len(dataset_test) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter evaluation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    # Build model
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        img_size=args.input_size,
    )
    
    # Load checkpoint
    if args.resume:
        print("Loading checkpoint from: %s" % args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        
        # Remove head if shape doesn't match
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from checkpoint")
                del checkpoint_model[k]
        
        # Interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)
        
        # Load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        
        # Check missing keys (more flexible for fine-tuned checkpoints)
        if args.global_pool:
            expected_missing = {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            # Fine-tuned checkpoints may already have these keys, so check if missing_keys is subset
            if msg.missing_keys:
                unexpected_missing = set(msg.missing_keys) - expected_missing
                if unexpected_missing:
                    print(f"Warning: Unexpected missing keys: {unexpected_missing}")
                    # Only warn, don't fail - checkpoint might be fine-tuned
        else:
            expected_missing = {'head.weight', 'head.bias'}
            if msg.missing_keys:
                unexpected_missing = set(msg.missing_keys) - expected_missing
                if unexpected_missing:
                    print(f"Warning: Unexpected missing keys: {unexpected_missing}")
        
        if msg.unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {msg.unexpected_keys}")
    
    model.to(device)
    
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    # Evaluate on test set
    print(f"Starting evaluation on test set ({len(dataset_test)} images)")
    start_time = time.time()
    
    test_stats = evaluate(data_loader_test, model, device)
    
    print(f"Test Results:")
    print(f"  Accuracy@1: {test_stats['acc1']:.2f}%")
    print(f"  Accuracy@5: {test_stats['acc5']:.2f}%")
    print(f"  Loss: {test_stats['loss']:.4f}")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time: {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
