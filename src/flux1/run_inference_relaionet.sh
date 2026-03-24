#!/bin/bash

conda activate flux2
set -e

DATA_PATH=/path/to/reLAIONet
NUM_SHARDS=4   # Set to number of available GPUs

run_config() {
    local STEPS=$1
    local CFG=$2
    local MODEL_NAME=$3
    local LAB_DIR=outputs/steps_${STEPS}_cfg${CFG}_relaionet

    mkdir -p "$LAB_DIR"
    local SHARD_DIR="$LAB_DIR/shards"
    mkdir -p "$SHARD_DIR"
    local CKPT_PATH="$LAB_DIR/generated_checkpoint.npz"

    echo "=== Generating: steps=$STEPS cfg=$CFG model=$MODEL_NAME ==="

    # Step 1: Launch one shard per GPU in parallel
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        CUDA_VISIBLE_DEVICES=$i python eval_relaionet.py \
            --data_path $DATA_PATH \
            --num_steps $STEPS \
            --guidance_scale $CFG \
            --model_name $MODEL_NAME \
            --shard_id $i \
            --num_shards $NUM_SHARDS \
            --shard_dir $SHARD_DIR &
    done
    wait
    echo "All shards complete."

    # Step 2: Combine shards into one checkpoint
    echo "Combining shards..."
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

    # Step 3: Compute metrics using --resume
    echo "=== Computing metrics: $MODEL_NAME ==="
    python eval_relaionet.py \
        --data_path $DATA_PATH \
        --num_steps $STEPS \
        --guidance_scale $CFG \
        --model_name $MODEL_NAME \
        --resume $CKPT_PATH
}

# Run 1: steps=1, cfg=7.0
run_config 1 7.0 flux1dev

# Run 2: steps=25, cfg=7.0
run_config 25 7.0 flux1dev

# Run 3: steps=1, cfg=3.5 (natural)
run_config 1 3.5 natural_flux1dev

# Run 4: steps=25, cfg=3.5 (natural)
run_config 25 3.5 natural_flux1dev
