#!/bin/bash
# MAE Fine-tuning script for Tiny ImageNet Classification (64x64, 200 classes)

# Set device ID (default: 0, can be overridden by command line argument)
DEVICE_ID=${1:-0}

# Set paths
DATA_DIR="/data/lhs1208/mae_res_64/data"
PRETRAIN_CHECKPOINT="/data/lhs1208/mae_res_64/output/pretrain/checkpoint-360.pth"  # Update with actual checkpoint path
OUTPUT_DIR="/data/lhs1208/mae_res_64/output/finetune"
LOG_DIR="/data/lhs1208/mae_res_64/logs/finetune"
MAE_DIR="/data/lhs1208/mae_res_64/mae"

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# Change to MAE directory
cd ${MAE_DIR}

# Set CUDA device
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

# Fine-tuning parameters
BATCH_SIZE=256
ACCUM_ITER=1
EPOCHS=100
WARMUP_EPOCHS=5
BLR=1e-3
WEIGHT_DECAY=0.05
LAYER_DECAY=0.75
INPUT_SIZE=64
MODEL="vit_base_patch4"
NB_CLASSES=200
NUM_WORKERS=8
DROP_PATH=0.1

# Augmentation parameters
COLOR_JITTER=0.4
AA="rand-m9-mstd0.5-inc1"
SMOOTHING=0.1
REPROB=0.25
REMODE="pixel"
RECOUNT=1

echo "Using GPU device: ${DEVICE_ID}"

# Run fine-tuning
python main_finetune.py \
    --batch_size ${BATCH_SIZE} \
    --accum_iter ${ACCUM_ITER} \
    --epochs ${EPOCHS} \
    --model ${MODEL} \
    --input_size ${INPUT_SIZE} \
    --drop_path ${DROP_PATH} \
    --blr ${BLR} \
    --layer_decay ${LAYER_DECAY} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --min_lr 1e-6 \
    --color_jitter ${COLOR_JITTER} \
    --aa ${AA} \
    --smoothing ${SMOOTHING} \
    --reprob ${REPROB} \
    --remode ${REMODE} \
    --recount ${RECOUNT} \
    --finetune ${PRETRAIN_CHECKPOINT} \
    --global_pool \
    --data_path ${DATA_DIR} \
    --nb_classes ${NB_CLASSES} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --num_workers ${NUM_WORKERS} \
    --pin_mem

echo "Fine-tuning completed. Checkpoints saved to: ${OUTPUT_DIR}"

