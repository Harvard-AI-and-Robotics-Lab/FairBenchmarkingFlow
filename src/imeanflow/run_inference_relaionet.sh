#!/bin/bash
set -e

# DATA_PATH=<SET_DATAPATH>
CHECKPOINT_PATH=weights/iMF-XL-2
NUM_GPUS=4

# Run 1: steps=1, omega=7.0
PYTORCH_ALLOC_CONF=expandable_segments:True python eval_relaionet.py \
    --data_path=$DATA_PATH \
    --checkpoint_path=$CHECKPOINT_PATH \
    --num_steps=1 \
    --num_gpus=$NUM_GPUS \
    --omega=7.0 \
    --model_str=MiT_XL_2 \
    --model_name=imeanflow_xl-2

# Run 2: steps=25, omega=7.0
PYTORCH_ALLOC_CONF=expandable_segments:True python eval_relaionet.py \
    --data_path=$DATA_PATH \
    --checkpoint_path=$CHECKPOINT_PATH \
    --num_steps=25 \
    --num_gpus=$NUM_GPUS \
    --omega=7.0 \
    --model_str=MiT_XL_2 \
    --model_name=imeanflow_xl-2

# Run 3: steps=1, omega=6.0 (natural)
PYTORCH_ALLOC_CONF=expandable_segments:True python eval_relaionet.py \
    --data_path=$DATA_PATH \
    --checkpoint_path=$CHECKPOINT_PATH \
    --num_steps=1 \
    --num_gpus=$NUM_GPUS \
    --omega=6.0 \
    --model_str=MiT_XL_2 \
    --model_name=natural_imeanflow_xl-2

# # Run 4: steps=25, omega=6.0 (natural)
PYTORCH_ALLOC_CONF=expandable_segments:True python eval_relaionet.py \
    --data_path=$DATA_PATH \
    --checkpoint_path=$CHECKPOINT_PATH \
    --num_steps=25 \
    --num_gpus=$NUM_GPUS \
    --omega=6.0 \
    --model_str=MiT_XL_2 \
    --model_name=natural_imeanflow_xl-2
