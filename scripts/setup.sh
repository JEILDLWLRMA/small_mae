#!/bin/bash
# Complete setup script for MAE training environment

set -e  # Exit on error

echo "=========================================="
echo "MAE Training Environment Setup"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found. Please run this script from the project root."
    exit 1
fi

# Step 1: Install dependencies
echo ""
echo "Step 1: Installing dependencies..."
echo "-----------------------------------"
pip install -r requirements.txt

# Step 2: Prepare dataset
echo ""
echo "Step 2: Preparing dataset..."
echo "-----------------------------------"
python scripts/prepare_tiny_imagenet.py

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "You can now run training scripts:"
echo "  bash scripts/train_pretrain.sh [gpu_id]"
echo "  bash scripts/train_finetune.sh [gpu_id]"
echo ""

