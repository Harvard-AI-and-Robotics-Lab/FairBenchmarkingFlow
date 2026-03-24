#!/bin/bash
set -e

DATA_PATH=/path/to/imagenet
CHECKPOINT_PATH=weights/SoFlow/XL-2-cond

PYTORCH_ALLOC_CONF=expandable_segments:True python eval_imagenet.py \
    --data_path=$DATA_PATH \
    --checkpoint_path=$CHECKPOINT_PATH \
    --num_steps=1 \
    --cfg=7.0 \
    --model_name=soflow_xl-2

PYTORCH_ALLOC_CONF=expandable_segments:True python eval_imagenet.py \
    --data_path=$DATA_PATH \
    --checkpoint_path=$CHECKPOINT_PATH \
    --num_steps=25 \
    --cfg=7.0 \
    --model_name=soflow_xl-2

PYTORCH_ALLOC_CONF=expandable_segments:True python eval_imagenet.py \
    --data_path=$DATA_PATH \
    --checkpoint_path=$CHECKPOINT_PATH \
    --num_steps=1 \
    --cfg=2 \
    --model_name=natural_soflow_xl-2

PYTORCH_ALLOC_CONF=expandable_segments:True python eval_imagenet.py \
    --data_path=$DATA_PATH \
    --checkpoint_path=$CHECKPOINT_PATH \
    --num_steps=25 \
    --cfg=2 \
    --model_name=natural_soflow_xl-2
