#!/bin/bash
set -e

CHECKPOINT_PATH=weights/iMF-XL-2
MODEL_STR=MiT_XL_2
NUM_SAMPLES=1
CLASSES=("goldfish" "groom" "scuba_diver" "golden_retriever" "sports_car")

# All (cfg_scale/omega, num_steps) settings from run_inference.sh
declare -a CFG_SCALES=(7.0  7.0  6.0  6.0)
declare -a NUM_STEPS=(1    25   1    25)

for i in "${!CFG_SCALES[@]}"; do
    CFG=${CFG_SCALES[$i]}
    STEPS=${NUM_STEPS[$i]}
    for CLASS in "${CLASSES[@]}"; do
        echo "==> cfg=${CFG}  steps=${STEPS}  class=${CLASS}"
        python eval_generate.py \
            --ckpt_path="${CHECKPOINT_PATH}" \
            --classname="${CLASS}" \
            --num_steps="${STEPS}" \
            --cfg_scale="${CFG}" \
            --model_str="${MODEL_STR}" \
            --num_samples="${NUM_SAMPLES}"
    done
done
