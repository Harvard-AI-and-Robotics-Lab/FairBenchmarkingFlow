#!/bin/bash

conda activate scale_rae
set -e

mkdir -p logs

DATA_PATH=/path/to/imagenet
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

    for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
        python eval_imagenet.py \
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

    python eval_imagenet.py \
        --data_path $DATA_PATH \
        --model_path $MODEL_PATH \
        --decoder_path $DECODER_PATH \
        --num_steps $STEPS \
        --guidance_level $GUIDANCE \
        --model_name $NAME \
        --device cuda:0
}

run_eval 25 7.0 scale_rae_s25_g7

echo "All runs complete."
