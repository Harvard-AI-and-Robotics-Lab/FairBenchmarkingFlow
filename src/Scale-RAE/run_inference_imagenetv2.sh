#!/bin/bash

conda activate scale_rae
set -e

DATA_PATH=/path/to/ImageNetV2
MODEL_PATH=models
DECODER_PATH=decoder/model.pt
NUM_GPUS=4

run_eval() {
    local STEPS=$1
    local GUIDANCE=$2
    local NAME=$3

    echo "============================================================"
    echo "  ${NAME} (steps=${STEPS}, guidance=${GUIDANCE})"
    echo "  Launching ${NUM_GPUS} parallel shards..."
    echo "============================================================"

    # Launch 4 generation processes in parallel
    for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
        python eval_imagenetv2.py \
            --data_path $DATA_PATH \
            --model_path $MODEL_PATH \
            --decoder_path $DECODER_PATH \
            --num_steps $STEPS \
            --guidance_level $GUIDANCE \
            --model_name $NAME \
            --device cuda:${GPU_ID} \
            --shard_id $GPU_ID \
            --num_shards $NUM_GPUS \
            --gen_only \
            &
    done
    wait

    echo "All shards done for ${NAME}. Merging + metrics..."

    # Merge shards and compute metrics (no --resume needed, auto-detects shards)
    python eval_imagenetv2.py \
        --data_path $DATA_PATH \
        --model_path $MODEL_PATH \
        --decoder_path $DECODER_PATH \
        --num_steps $STEPS \
        --guidance_level $GUIDANCE \
        --model_name $NAME \
        --device cuda:0
}

# Run 1: 1 step, guidance=7.0
#run_eval 1 7.0 scale_rae_s1_g7

# Run 2: 25 steps, guidance=7.0
#run_eval 25 7.0 scale_rae_s25_g7

# Run 3: 1 step, guidance=1.42 (natural)
#run_eval 1 1.42 natural_scale_rae_s1_g1.42

# Run 4: 25 steps, guidance=1.42 (natural)
run_eval 25 1.42 natural_scale_rae_s25_g1.42

echo "All runs complete."
