#!/bin/bash

# Fine-tune MeanFlow on Mini-ImageNet
# Usage: bash scripts/finetune_miniimagenet.sh [optional_job_suffix]

# ============================================
# Configuration — update these paths
# ============================================
export DATA_PATH="/path/to/mini_imagenet"
export PRETRAINED_CKPT="weights/checkpoint_1200960/"
export LOG_DIR="logs"

# Validate
if [ "$DATA_PATH" = "YOUR_MINI_IMAGENET_PATH" ] || [ "$PRETRAINED_CKPT" = "YOUR_PRETRAINED_CHECKPOINT_PATH" ] || [ "$LOG_DIR" = "YOUR_LOG_DIR" ]; then
    echo "ERROR: Please update the paths at the top of this script:"
    echo "  - DATA_PATH: Path to Mini-ImageNet root (with train.csv + images/)"
    echo "  - PRETRAINED_CKPT: Path to pretrained MeanFlow checkpoint directory"
    echo "  - LOG_DIR: Path for output logs and checkpoints"
    exit 1
fi

# ============================================
# Hyperparameters
# ============================================
MODEL_CLS=${MODEL_CLS:-DiT_B_4}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_EPOCHS=${NUM_EPOCHS:-100}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
VAE_TYPE=${VAE_TYPE:-mse}
SEED=${SEED:-42}
WANDB_PROJECT=${WANDB_PROJECT:-meanflow_reimplementation}

# ============================================
# Job setup
# ============================================
export now=$(date '+%Y%m%d_%H%M%S')
export salt=$(head /dev/urandom | tr -dc a-z0-9 | head -c6)
export JOBNAME=finetune_miniimagenet_${now}_${salt}${1:+_$1}
export WORKDIR=$LOG_DIR/$USER/$JOBNAME

mkdir -p ${WORKDIR}

echo "=============================================="
echo "MeanFlow Mini-ImageNet Fine-Tuning"
echo "=============================================="
echo "Data Path:        $DATA_PATH"
echo "Pretrained Ckpt:  $PRETRAINED_CKPT"
echo "Work Dir:         $WORKDIR"
echo "Model:            $MODEL_CLS"
echo "Batch Size:       $BATCH_SIZE"
echo "Epochs:           $NUM_EPOCHS"
echo "Learning Rate:    $LEARNING_RATE"
echo "VAE Type:         $VAE_TYPE"
echo "WandB Project:    $WANDB_PROJECT"
echo "=============================================="

python3 finetune_miniimagenet.py \
    --data_path=${DATA_PATH} \
    --pretrained_ckpt=${PRETRAINED_CKPT} \
    --workdir=${WORKDIR} \
    --model_cls=${MODEL_CLS} \
    --batch_size=${BATCH_SIZE} \
    --num_epochs=${NUM_EPOCHS} \
    --learning_rate=${LEARNING_RATE} \
    --vae_type=${VAE_TYPE} \
    --seed=${SEED} \
    --wandb_project=${WANDB_PROJECT} \
    2>&1 | tee -a $WORKDIR/output.log

echo "=============================================="
echo "Fine-tuning completed!"
echo "Logs and checkpoints at: $WORKDIR"
echo "=============================================="
