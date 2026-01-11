#!/bin/bash
# MAE Test script for Tiny ImageNet Classification (64x64, 200 classes)

# Set device ID (default: 0, can be overridden by command line argument)
DEVICE_ID=${1:-0}

# Set paths
DATA_DIR="/data/lhs1208/mae_res_64/data"
CHECKPOINT=${2:-"/data/lhs1208/mae_res_64/output/finetune/checkpoint-99.pth"}  # Default to last checkpoint
MAE_DIR="/data/lhs1208/mae_res_64/mae"

# Change to MAE directory
cd ${MAE_DIR}

# Set CUDA device
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

# Test parameters
BATCH_SIZE=256
INPUT_SIZE=64
MODEL="vit_base_patch4"
NB_CLASSES=200
NUM_WORKERS=8

echo "Using GPU device: ${DEVICE_ID}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Test dataset: ${DATA_DIR}/test"

# Run test
python main_test.py \
    --batch_size ${BATCH_SIZE} \
    --model ${MODEL} \
    --input_size ${INPUT_SIZE} \
    --nb_classes ${NB_CLASSES} \
    --data_path ${DATA_DIR} \
    --resume ${CHECKPOINT} \
    --num_workers ${NUM_WORKERS} \
    --pin_mem \
    --global_pool

echo "Test completed!"
