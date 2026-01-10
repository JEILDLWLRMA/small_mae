# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # For small images (64x64), use simpler augmentation
        if args.input_size <= 64:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=PIL.Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation='bicubic',
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=mean,
                std=std,
            )
        return transform

    # eval transform
    t = []
    # For small images, no need for resizing/cropping if already correct size
    if args.input_size <= 64:
        # Images are already 64x64, just normalize
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
    elif args.input_size <= 224:
        crop_pct = 224 / 256
        size = int(args.input_size / crop_pct)
        t.append(
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
    else:
        crop_pct = 1.0
        size = int(args.input_size / crop_pct)
        t.append(
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
        )
        t.append(transforms.CenterCrop(args.input_size))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
