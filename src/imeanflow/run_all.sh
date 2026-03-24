#!/bin/bash
set -e

DATA_PATH=/path/to/imagenet
CHECKPOINT_PATH=weights/iMF-XL-2
NUM_GPUS=4
OUTPUT_DIR=outputs/imeanflow/ablations

CFG_VALUES=(1 3 6 7 9 12 15)
STEP_VALUES=(1 5 10 15 20 25)

for CFG in "${CFG_VALUES[@]}"; do
    for STEPS in "${STEP_VALUES[@]}"; do
        CSV_PATH="../../${OUTPUT_DIR}/cfg${CFG}_steps_${STEPS}_imagenet/cfg${CFG}_steps_${STEPS}_imagenet.csv"
        if [ -f "$CSV_PATH" ]; then
            echo "Skipping: cfg=${CFG}, steps=${STEPS} (CSV exists: ${CSV_PATH})"
            continue
        fi
        echo "=========================================="
        echo "Running: cfg=${CFG}, steps=${STEPS}"
        echo "=========================================="
        PYTORCH_ALLOC_CONF=expandable_segments:True python eval_imagenet.py \
            --data_path=$DATA_PATH \
            --checkpoint_path=$CHECKPOINT_PATH \
            --num_steps=$STEPS \
            --num_gpus=$NUM_GPUS \
            --omega=$CFG \
            --model_str=MiT_XL_2 \
            --model_name=cfg${CFG} \
            --output_dir=$OUTPUT_DIR \
            --resume || {
            # JAX/XLA CUDA cleanup can exit non-zero even after a successful run.
            # Treat as success if the CSV was written; abort on real failures.
            if [ ! -f "$CSV_PATH" ]; then
                echo "ERROR: cfg=${CFG}, steps=${STEPS} failed (no CSV written)" >&2
                exit 1
            fi
            echo "WARNING: Non-zero exit for cfg=${CFG}, steps=${STEPS} — likely XLA CUDA cleanup noise; CSV exists, continuing."
        }
    done
done
