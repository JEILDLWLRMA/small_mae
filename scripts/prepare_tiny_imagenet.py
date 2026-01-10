#!/usr/bin/env python3
"""
Download and prepare Tiny ImageNet dataset from Hugging Face.
Converts to ImageFolder structure and splits train data 90/10.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def download_and_organize_dataset(data_dir: str):
    """
    Download tiny-imagenet from Hugging Face and organize into ImageFolder structure.
    
    Args:
        data_dir: Root directory where data will be stored
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading tiny-imagenet dataset from Hugging Face...")
    dataset = load_dataset('zh-plus/tiny-imagenet')
    
    # Get train and valid splits
    train_split = dataset['train']
    valid_split = dataset['valid']
    
    print(f"Train split: {len(train_split)} images")
    print(f"Valid split: {len(valid_split)} images")
    
    # Create directory structure
    train_dir = data_path / 'train'
    val_dir = data_path / 'val'
    test_dir = data_path / 'test'
    
    # Remove existing directories if they exist
    for d in [train_dir, val_dir, test_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    
    # First, organize valid split as test (for leaderboard)
    print("\nOrganizing valid split as test...")
    class_to_idx = {}
    idx_to_class = {}
    idx = 0
    
    # Collect all unique labels from train to build class mapping
    train_labels = set()
    for item in train_split:
        label = item['label']
        train_labels.add(label)
    
    # Build class mapping
    sorted_labels = sorted(train_labels)
    for label in sorted_labels:
        class_to_idx[label] = idx
        idx_to_class[idx] = label
        idx += 1
    
    print(f"Found {len(class_to_idx)} classes")
    
    # Save test split (from valid)
    print("\nSaving test split (from valid)...")
    test_class_dirs = {}
    for label in sorted_labels:
        test_class_dir = test_dir / str(label)
        test_class_dir.mkdir(exist_ok=True)
        test_class_dirs[label] = test_class_dir
    
    test_counters = {label: 0 for label in sorted_labels}
    for item in tqdm(valid_split, desc="Processing valid split"):
        label = item['label']
        image = item['image']
        
        # Save image
        image_path = test_class_dirs[label] / f"{test_counters[label]:05d}.jpg"
        image.save(image_path)
        test_counters[label] += 1
    
    # Organize train split and prepare for stratified split
    print("\nOrganizing train split...")
    train_class_dirs = {}
    train_data_by_class = defaultdict(list)
    
    for label in sorted_labels:
        train_class_dir = train_dir / str(label)
        train_class_dir.mkdir(exist_ok=True)
        train_class_dirs[label] = train_class_dir
    
    # Collect all train data organized by class
    for item in tqdm(train_split, desc="Processing train split"):
        label = item['label']
        train_data_by_class[label].append(item)
    
    # Perform stratified split (90/10) for each class
    print("\nPerforming stratified 90/10 split...")
    val_class_dirs = {}
    for label in sorted_labels:
        val_class_dir = val_dir / str(label)
        val_class_dir.mkdir(exist_ok=True)
        val_class_dirs[label] = val_class_dir
    
    total_train = 0
    total_val = 0
    
    for label in sorted_labels:
        items = train_data_by_class[label]
        indices = np.arange(len(items))
        
        # Stratified split maintaining class distribution
        train_indices, val_indices = train_test_split(
            indices, test_size=0.1, random_state=42, shuffle=True
        )
        
        # Save train images
        train_counter = 0
        for idx in train_indices:
            item = items[idx]
            image = item['image']
            image_path = train_class_dirs[label] / f"{train_counter:05d}.jpg"
            image.save(image_path)
            train_counter += 1
            total_train += 1
        
        # Save val images
        val_counter = 0
        for idx in val_indices:
            item = items[idx]
            image = item['image']
            image_path = val_class_dirs[label] / f"{val_counter:05d}.jpg"
            image.save(image_path)
            val_counter += 1
            total_val += 1
    
    print(f"\nDataset organization complete!")
    print(f"Train: {total_train} images")
    print(f"Val: {total_val} images")
    print(f"Test: {len(valid_split)} images")
    print(f"Total classes: {len(class_to_idx)}")
    
    # Save class mapping
    class_mapping_file = data_path / 'class_mapping.txt'
    with open(class_mapping_file, 'w') as f:
        for idx in sorted(idx_to_class.keys()):
            f.write(f"{idx}\t{idx_to_class[idx]}\n")
    
    print(f"\nClass mapping saved to: {class_mapping_file}")
    print(f"\nData structure:")
    print(f"  {train_dir}/")
    print(f"  {val_dir}/")
    print(f"  {test_dir}/")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare Tiny ImageNet dataset')
    parser.add_argument('--data_dir', type=str, default='/data/lhs1208/mae_res_64/data',
                        help='Directory to save the dataset')
    
    args = parser.parse_args()
    download_and_organize_dataset(args.data_dir)

