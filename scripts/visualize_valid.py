#!/usr/bin/env python3
"""
Visualize MAE reconstructions on validation images.
Loads the most recent checkpoint from output/pretrain and visualizes random images from data/val.
"""

import os
import sys
import glob
import random
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Add mae directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mae'))
import models_mae

# ImageNet normalization constants
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint file."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint-*.pth'))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Sort by modification time (most recent first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    latest_checkpoint = checkpoint_files[0]
    
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def load_model(checkpoint_path, arch='mae_vit_base_patch4', img_size=64, device='cuda'):
    """Load MAE model from checkpoint."""
    print(f"Loading model: {arch}")
    model = models_mae.__dict__[arch](img_size=img_size, norm_pix_loss=False)
    
    # Load checkpoint with weights_only=False for PyTorch 2.6+ compatibility
    # (checkpoints may contain argparse.Namespace objects)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Model loading message: {msg}")
    
    model.to(device)
    model.eval()
    return model


def denormalize_image(img_tensor):
    """Denormalize ImageNet-normalized image tensor."""
    # img_tensor: [C, H, W] or [H, W, C]
    if len(img_tensor.shape) == 3 and img_tensor.shape[0] == 3:
        # [C, H, W] -> [H, W, C]
        img_tensor = img_tensor.permute(1, 2, 0)
    
    img_np = img_tensor.cpu().numpy()
    img_denorm = img_np * imagenet_std + imagenet_mean
    img_denorm = np.clip(img_denorm, 0, 1)
    return img_denorm


def visualize_one_image(img, model, mask_ratio=0.75, device='cuda'):
    """
    Run MAE on a single image and return visualization components.
    
    Args:
        img: PIL Image or numpy array (normalized)
        model: MAE model
        mask_ratio: masking ratio
        device: device to run on
    
    Returns:
        Dictionary with visualization components
    """
    # Convert PIL to tensor if needed
    if isinstance(img, Image.Image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
        img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    elif isinstance(img, np.ndarray):
        img_tensor = torch.from_numpy(img).float()
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        if img_tensor.shape[1] != 3:  # [H, W, C] -> [C, H, W]
            img_tensor = img_tensor.permute(0, 3, 1, 2)
    else:
        # img is already a tensor (from ImageFolder)
        img_tensor = img
        # Ensure it has batch dimension [B, C, H, W]
        if len(img_tensor.shape) == 3:  # [C, H, W]
            img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
    
    img_tensor = img_tensor.to(device)
    
    # Run MAE
    with torch.no_grad():
        loss, y, mask = model(img_tensor, mask_ratio=mask_ratio)
        y = model.unpatchify(y)
        y = y.detach().cpu()
    
    # Process mask for visualization
    mask = mask.detach().cpu()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = mask.detach().cpu()
    
    # Convert to numpy for visualization
    img_original = img_tensor.cpu()
    img_masked = img_original * (1 - mask)
    img_reconstruction = y
    img_paste = img_original * (1 - mask) + y * mask
    
    # Denormalize all images
    img_original_np = denormalize_image(img_original[0])
    img_masked_np = denormalize_image(img_masked[0])
    img_reconstruction_np = denormalize_image(img_reconstruction[0])
    img_paste_np = denormalize_image(img_paste[0])
    
    return {
        'original': img_original_np,
        'masked': img_masked_np,
        'reconstruction': img_reconstruction_np,
        'reconstruction_visible': img_paste_np,
        'loss': loss.item()
    }


def save_visualization(vis_dict, save_path, title_prefix=''):
    """Save visualization as a figure with 4 subplots."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(vis_dict['original'])
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(vis_dict['masked'])
    axes[1].set_title('Masked', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(vis_dict['reconstruction'])
    axes[2].set_title('Reconstruction', fontsize=14)
    axes[2].axis('off')
    
    axes[3].imshow(vis_dict['reconstruction_visible'])
    axes[3].set_title('Reconstruction + Visible', fontsize=14)
    axes[3].axis('off')
    
    if title_prefix:
        fig.suptitle(f'{title_prefix} (Loss: {vis_dict["loss"]:.4f})', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser('MAE Visualization on Validation Set')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='/data/lhs1208/mae_res_64/output/pretrain',
                       help='Directory containing checkpoint files')
    parser.add_argument('--val_dir', type=str,
                       default='/data/lhs1208/mae_res_64/data/val',
                       help='Directory containing validation images')
    parser.add_argument('--output_dir', type=str,
                       default='/data/lhs1208/mae_res_64/output/pretrain/visualize',
                       help='Directory to save visualizations')
    parser.add_argument('--num_images', type=int, default=10,
                       help='Number of random images to visualize')
    parser.add_argument('--model', type=str, default='mae_vit_base_patch4',
                       help='Model architecture')
    parser.add_argument('--input_size', type=int, default=64,
                       help='Input image size')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                       help='Masking ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, cpu, cuda:0, cuda:1, etc.). If not specified, uses --gpu option.')
    parser.add_argument('--gpu', type=int, default=6,
                       help='GPU device ID to use (default: 0). Use -1 for CPU.')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Find latest checkpoint
    checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
    
    # Determine device
    if args.device is not None:
        # Use explicit device string
        if args.device.isdigit():
            # If just a number, interpret as GPU ID
            device = torch.device(f'cuda:{args.device}')
        else:
            device = torch.device(args.device)
    else:
        # Use --gpu option
        if args.gpu < 0:
            device = torch.device('cpu')
        else:
            if torch.cuda.is_available():
                if args.gpu >= torch.cuda.device_count():
                    print(f"Warning: GPU {args.gpu} not available. Using GPU 0 instead.")
                    device = torch.device('cuda:0')
                else:
                    device = torch.device(f'cuda:{args.gpu}')
            else:
                print("Warning: CUDA not available. Using CPU instead.")
                device = torch.device('cpu')
    
    print(f"Using device: {device}")
    model = load_model(checkpoint_path, arch=args.model, img_size=args.input_size, device=device)
    
    # Load validation dataset
    print(f"Loading validation dataset from {args.val_dir}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    val_dataset = datasets.ImageFolder(args.val_dir, transform=transform)
    
    # Randomly select images
    num_images = min(args.num_images, len(val_dataset))
    selected_indices = random.sample(range(len(val_dataset)), num_images)
    print(f"Selected {num_images} random images for visualization")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving visualizations to {args.output_dir}")
    
    # Visualize each image
    for idx, sample_idx in enumerate(selected_indices):
        img_tensor, label = val_dataset[sample_idx]
        
        # Get original image path for filename
        img_path = val_dataset.samples[sample_idx][0]
        img_name = Path(img_path).stem
        
        # Get class name
        class_name = val_dataset.classes[label]
        
        # Run visualization
        print(f"Processing image {idx+1}/{num_images}: {img_name} (class: {class_name})")
        vis_dict = visualize_one_image(
            img_tensor, model, mask_ratio=args.mask_ratio, device=device
        )
        
        # Save visualization
        save_path = os.path.join(args.output_dir, f'{idx+1:03d}_{img_name}_class{class_name}.png')
        save_visualization(vis_dict, save_path, title_prefix=f'{img_name} (class: {class_name})')
    
    print(f"\nVisualization complete! Saved {num_images} images to {args.output_dir}")


if __name__ == '__main__':
    main()

