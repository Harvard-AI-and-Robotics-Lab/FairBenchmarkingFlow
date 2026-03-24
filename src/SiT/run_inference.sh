#!/bin/bash
set -e

DATA_PATH=/path/to/imagenet
CKPT_PATH=pre_trained/SiT-XL-2-256x256.pt
CFG_NAT=1.5

PYTORCH_ALLOC_CONF=expandable_segments:True python eval_imagenet.py \
    --data-path=$DATA_PATH \
    --ckpt=$CKPT_PATH \
    --cfg=7.0 \
    --num-steps=25 \
    --model-name=sit_xl_2

PYTORCH_ALLOC_CONF=expandable_segments:True python eval_imagenet.py \
    --data-path=$DATA_PATH \
    --ckpt=$CKPT_PATH \
    --cfg=$CFG_NAT= \
    --num-steps=25 \
    --model-name=natural_sit_xl_2
