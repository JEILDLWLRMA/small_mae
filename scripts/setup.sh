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

# Step 2: Apply timm compatibility patch
echo ""
echo "Step 2: Applying timm compatibility patch..."
echo "-----------------------------------"
if python scripts/fix_timm_compatibility.py; then
    echo "✓ Compatibility patch applied successfully"
else
    echo "✗ Failed to apply patch. Please check the error above."
    exit 1
fi

# Step 3: Verify installation
echo ""
echo "Step 3: Verifying installation..."
echo "-----------------------------------"
python -c "import torch; import timm; print(f'✓ PyTorch {torch.__version__}')"
python -c "import timm; print(f'✓ timm {timm.__version__}')"

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "You can now run training scripts:"
echo "  bash scripts/train_pretrain.sh [gpu_id]"
echo "  bash scripts/train_finetune.sh [gpu_id]"
echo ""

