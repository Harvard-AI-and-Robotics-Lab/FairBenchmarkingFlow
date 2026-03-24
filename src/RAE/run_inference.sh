#!/bin/bash
set -e

DATA_PATH=/path/to/imagenet
CONFIG=configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml
NUM_GPUS=4

# Run 1: steps=1, cfg=7.0
torchrun --standalone --nproc_per_node=$NUM_GPUS eval_imagenet.py \
    --config $CONFIG \
    --data_path $DATA_PATH \
    --num_steps 1 \
    --cfg_scale 7.0 \
    --model_name rae

# Run 2: steps=25, cfg=7.0
torchrun --standalone --nproc_per_node=$NUM_GPUS eval_imagenet.py \
    --config $CONFIG \
    --data_path $DATA_PATH \
    --num_steps 25 \
    --cfg_scale 7.0 \
    --model_name rae

# Run 3: steps=1, cfg=1.42 (natural)
torchrun --standalone --nproc_per_node=$NUM_GPUS eval_imagenet.py \
    --config $CONFIG \
    --data_path $DATA_PATH \
    --num_steps 1 \
    --cfg_scale 1.42 \
    --model_name natural_rae

# Run 4: steps=25, cfg=1.42 (natural)
torchrun --standalone --nproc_per_node=$NUM_GPUS eval_imagenet.py \
    --config $CONFIG \
    --data_path $DATA_PATH \
    --num_steps 25 \
    --cfg_scale 1.42 \
    --model_name natural_rae
