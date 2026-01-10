#!/bin/bash
# MAE Pre-training script for Tiny ImageNet (64x64)

# Set device ID (default: 0, can be overridden by command line argument)
DEVICE_ID=${1:-0}

# Set paths
DATA_DIR="/data/lhs1208/mae_res_64/data"
OUTPUT_DIR="/data/lhs1208/mae_res_64/output/pretrain"
LOG_DIR="/data/lhs1208/mae_res_64/logs/pretrain"
MAE_DIR="/data/lhs1208/mae_res_64/mae"

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# Change to MAE directory
cd ${MAE_DIR}

# Set CUDA device
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

# Pre-training parameters
BATCH_SIZE=256
ACCUM_ITER=1
EPOCHS=400
WARMUP_EPOCHS=40
BLR=1.5e-4
WEIGHT_DECAY=0.05
MASK_RATIO=0.75
INPUT_SIZE=64
MODEL="mae_vit_base_patch4"
NUM_WORKERS=8

echo "Using GPU device: ${DEVICE_ID}"

# Run pre-training
python main_pretrain.py \
    --batch_size ${BATCH_SIZE} \
    --accum_iter ${ACCUM_ITER} \
    --epochs ${EPOCHS} \
    --model ${MODEL} \
    --input_size ${INPUT_SIZE} \
    --mask_ratio ${MASK_RATIO} \
    --norm_pix_loss \
    --blr ${BLR} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --data_path ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --num_workers ${NUM_WORKERS} \
    --pin_mem

echo "Pre-training completed. Checkpoints saved to: ${OUTPUT_DIR}"

