#!/bin/bash

conda activate flux2 
set -e

DATA_PATH=/path/to/imagenet
TOTAL=50000
NUM_SHARDS=4   # Set to number of available GPUs
PER_SHARD=$((TOTAL / NUM_SHARDS))

run_config() {
    local STEPS=$1
    local CFG=$2
    local MODEL_NAME=$3
    local OUTPUT_DIR=outputs/steps_${STEPS}_cfg${CFG}

    mkdir -p "$OUTPUT_DIR"

    echo "=== Generating: steps=$STEPS cfg=$CFG model=$MODEL_NAME ==="

    # Launch one shard per GPU in parallel
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        CUDA_VISIBLE_DEVICES=$i python eval_imagenet_shard.py \
            --data_path $DATA_PATH \
            --num_samples $PER_SHARD \
            --num_steps $STEPS \
            --guidance_scale $CFG \
            --seed $((42 + i)) \
            --shard_id $i \
            --num_shards $NUM_SHARDS \
            --output_dir "$OUTPUT_DIR" \
            --ckpt_every 25 &
    done
    wait

    echo "=== Computing metrics: $MODEL_NAME ==="
    python eval_imagenet_combine.py \
        --data_path $DATA_PATH \
        --shard_dir "$OUTPUT_DIR" \
        --num_shards $NUM_SHARDS \
        --num_steps $STEPS \
        --guidance_scale $CFG \
        --model_name $MODEL_NAME
}

# Run 1: steps=1, cfg=7.0
run_config 1 7.0 flux1dev

# Run 2: steps=25, cfg=7.0
run_config 25 7.0 flux1dev

# Run 3: steps=1, cfg=3.5 (natural)
run_config 1 3.5 natural_flux1dev

# Run 4: steps=25, cfg=3.5 (natural)
run_config 25 3.5 natural_flux1dev
