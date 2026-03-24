#!/bin/bash
set -e

DATA_PATH=/path/to/imagenet
CHECKPOINT_PATH=weights/checkpoint_1200960/
NUM_GPUS=4

PYTORCH_ALLOC_CONF=expandable_segments:True python eval_imagenet.py \
    --data_path=$DATA_PATH \
    --checkpoint_path=$CHECKPOINT_PATH \
    --num_steps=1 \
    --num_gpus=$NUM_GPUS \
    --omega=7.0 \
    --model_name=meanflow_b-4

PYTORCH_ALLOC_CONF=expandable_segments:True python eval_imagenet.py \
    --data_path=$DATA_PATH \
    --checkpoint_path=$CHECKPOINT_PATH \
    --num_steps=25 \
    --num_gpus=$NUM_GPUS \
    --omega=7.0 \
    --model_name=meanflow_b-4

PYTORCH_ALLOC_CONF=expandable_segments:True python eval_imagenet.py \
    --data_path=$DATA_PATH \
    --checkpoint_path=$CHECKPOINT_PATH \
    --num_steps=1 \
    --num_gpus=$NUM_GPUS \
    --omega=1.0 \
    --model_name=natural_meanflow_b-4

PYTORCH_ALLOC_CONF=expandable_segments:True python eval_imagenet.py \
    --data_path=$DATA_PATH \
    --checkpoint_path=$CHECKPOINT_PATH \
    --num_steps=25 \
    --num_gpus=$NUM_GPUS \
    --omega=1.0 \
    --model_name=natural_meanflow_b-4
