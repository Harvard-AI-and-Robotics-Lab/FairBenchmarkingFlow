#!/bin/bash

conda activate sd35
set -e

DATA_PATH=/path/to/imagenet
NUM_SHARDS=4
NUM_STEPS=25
GUIDANCE=3.5
MODEL=natural_sd35l

# ============================================================
# Output directory (write directly here; no symlinks)
# ============================================================
LAB_DIR=outputs/sd35_natural_25steps

# Create logs folder relative to where you submit from (or make it absolute)
mkdir -p logs
mkdir -p "$LAB_DIR"

SHARD_DIR="$LAB_DIR/shards"
mkdir -p "$SHARD_DIR"
# ============================================================
# Step 1: Generate images in parallel (one GPU per shard)
# ============================================================
echo "Launching $NUM_SHARDS shards across GPUs..."

for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    echo "  Starting shard $SHARD_ID on GPU $SHARD_ID"
    CUDA_VISIBLE_DEVICES=$SHARD_ID python eval_imagenet.py \
        --data_path $DATA_PATH \
        --num_steps $NUM_STEPS \
        --guidance_scale $GUIDANCE \
        --model_name $MODEL \
        --shard_id $SHARD_ID \
        --num_shards $NUM_SHARDS \
        --shard_dir $SHARD_DIR &
done

wait
echo "All shards complete."

# ============================================================
# Step 2: Combine shards into one checkpoint
# ============================================================
echo "Combining shards..."

CKPT_PATH=$LAB_DIR/generated_checkpoint.npz

python -c "
import numpy as np, os, sys
shard_dir = '${SHARD_DIR}'
num_shards = ${NUM_SHARDS}
all_images, all_labels = [], []
for i in range(num_shards):
    path = os.path.join(shard_dir, f'shard_{i}.npz')
    if not os.path.exists(path):
        print(f'ERROR: missing {path}'); sys.exit(1)
    data = np.load(path)
    all_images.append(data['images'])
    all_labels.append(data['labels'])
    print(f'  Shard {i}: {len(data[\"images\"])} images')
images = np.concatenate(all_images)
labels = np.concatenate(all_labels)
out = '${CKPT_PATH}'
np.savez_compressed(out, images=images, labels=labels)
print(f'Combined: {images.shape} -> {out}')
"

echo "Checkpoint saved to $CKPT_PATH"

# ============================================================
# Step 3: Run metrics using --resume
# (CSV + PNG land in $LAB_DIR via the symlink)
# ============================================================
echo "Running metrics..."

python eval_imagenet.py \
    --data_path $DATA_PATH \
    --num_steps $NUM_STEPS \
    --guidance_scale $GUIDANCE \
    --model_name $MODEL \
    --resume $CKPT_PATH

echo "Done! All outputs in: $LAB_DIR"