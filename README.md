# Tiny ImageNet MAE Training

This repository contains scripts and configurations for training MAE (Masked Autoencoder) on Tiny ImageNet dataset (64x64 images, 200 classes).

## Dataset Structure

The dataset is organized in ImageFolder format:
```
/data/lhs1208/mae_res_64/data/
├── train/          # 90% of original train (90,000 images)
│   ├── 0/          # Class directories (0-199)
│   ├── 1/
│   └── ...
├── val/            # 10% of original train (10,000 images) - for validation during training
│   ├── 0/
│   ├── 1/
│   └── ...
└── test/           # Original valid split (10,000 images) - for leaderboard/testing
    ├── 0/
    ├── 1/
    └── ...
```

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Apply timm compatibility patch** (required for PyTorch >= 1.9.0):
```bash
# This fixes the torch._six import error in timm 0.3.2
python scripts/fix_timm_compatibility.py
```

**Important**: 
- The compatibility patch **must** be applied after installing timm
- If you see "Could not import timm", install dependencies first (`pip install -r requirements.txt`)
- The patch fixes the `ModuleNotFoundError: No module named 'torch._six'` error automatically

2. **Prepare the dataset**:
```bash
cd /data/lhs1208/mae_res_64
python scripts/prepare_tiny_imagenet.py --data_dir /data/lhs1208/mae_res_64/data
```

This will:
- Download tiny-imagenet from Hugging Face
- Organize into ImageFolder structure
- Split train data 90/10 (stratified)
- Save valid split as test

## Model Architecture

- **Pre-training**: `mae_vit_base_patch4` - ViT-Base with patch size 4 (for 64x64 images)
- **Fine-tuning**: `vit_base_patch4` - ViT-Base with patch size 4, 200 classes

Patch size 4 means 64x64 images are divided into 16x16 patches (64/4 = 16).

## Training

### Pre-training

Train MAE from scratch:
```bash
# Use default GPU (device 0)
bash scripts/train_pretrain.sh

# Or specify GPU device ID
bash scripts/train_pretrain.sh 1  # Use GPU 1
```

Or run manually (with device selection):
```bash
cd /data/lhs1208/mae_res_64/mae
# Set CUDA_VISIBLE_DEVICES to select GPU
CUDA_VISIBLE_DEVICES=0 python main_pretrain.py \
    --batch_size 256 \
    --epochs 400 \
    --model mae_vit_base_patch4 \
    --input_size 64 \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --warmup_epochs 40 \
    --data_path /data/lhs1208/mae_res_64/data \
    --output_dir /data/lhs1208/mae_res_64/output/pretrain \
    --log_dir /data/lhs1208/mae_res_64/logs/pretrain
```

### Fine-tuning

Fine-tune for classification (after pre-training):
```bash
# Use default GPU (device 0)
bash scripts/train_finetune.sh

# Or specify GPU device ID
bash scripts/train_finetune.sh 1  # Use GPU 1
```

Make sure to update the `PRETRAIN_CHECKPOINT` path in the script first!

Or run manually (with device selection):
```bash
cd /data/lhs1208/mae_res_64/mae
# Set CUDA_VISIBLE_DEVICES to select GPU
CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
    --batch_size 128 \
    --epochs 100 \
    --model vit_base_patch4 \
    --input_size 64 \
    --nb_classes 200 \
    --drop_path 0.1 \
    --blr 1e-3 \
    --layer_decay 0.75 \
    --weight_decay 0.05 \
    --warmup_epochs 5 \
    --finetune /data/lhs1208/mae_res_64/output/pretrain/checkpoint-399.pth \
    --global_pool \
    --data_path /data/lhs1208/mae_res_64/data \
    --output_dir /data/lhs1208/mae_res_64/output/finetune \
    --log_dir /data/lhs1208/mae_res_64/logs/finetune
```

## Configuration Files

- `configs/pretrain_config.yaml` - Pre-training hyperparameters
- `configs/finetune_config.yaml` - Fine-tuning hyperparameters

## Output Structure

```
/data/lhs1208/mae_res_64/
├── output/
│   ├── pretrain/    # Pre-training checkpoints
│   └── finetune/    # Fine-tuning checkpoints
└── logs/
    ├── pretrain/    # Pre-training tensorboard logs
    └── finetune/    # Fine-tuning tensorboard logs
```

## Notes

- The valid split from the original dataset is saved as `test/` for future leaderboard evaluation
- Train data is split 90/10 for training and validation during fine-tuning
- Pre-training uses only the train split (no labels needed)
- Fine-tuning uses train/val splits for supervised classification

## Acknowledgments

This repository is based on the [MAE (Masked Autoencoders)](https://github.com/facebookresearch/mae) implementation by Facebook Research, adapted for Tiny ImageNet (64x64 images).

**Original Paper**:  
[Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)  
Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick

**Original Repository**:  
https://github.com/facebookresearch/mae

**Modifications**:
- Adapted for 64x64 image size (Tiny ImageNet)
- Fixed compatibility issues with latest PyTorch and timm versions
- Added training scripts and configurations for Tiny ImageNet

## License

This project follows the **CC-BY-NC 4.0** license from the original MAE repository. See [LICENSE](LICENSE) for details.

**Important**: This license allows non-commercial use only. For commercial use, please refer to the original MAE repository and its licensing terms.

