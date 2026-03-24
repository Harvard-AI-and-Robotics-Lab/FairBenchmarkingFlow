#!/bin/bash

conda activate flux2
set -e

DATA_PATH=/path/to/imagenet
TOTAL=50000
NUM_SHARDS=8   # 2 nodes x 4 GPUs
PER_SHARD=$((TOTAL / NUM_SHARDS))

mkdir -p logs

STEPS=25
CFG=7.0
MODEL_NAME=flux1dev
OUTPUT_DIR=outputs/steps_${STEPS}_cfg${CFG}
mkdir -p "$OUTPUT_DIR"

echo "=== Generating: steps=$STEPS cfg=$CFG shards=$NUM_SHARDS ==="
echo "=== Nodes: $SLURM_JOB_NODELIST ==="

# Get list of allocated nodes
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
echo "Node 0: ${NODES[0]}"
echo "Node 1: ${NODES[1]}"

# Launch 4 shards on each node (8 total)
SHARD_ID=0
for NODE_IDX in 0 1; do
    for GPU_ID in 0 1 2 3; do
        srun --nodes=1 --ntasks=1 --gpus=1 --nodelist=${NODES[$NODE_IDX]} \
            bash -c "
                CUDA_VISIBLE_DEVICES=$GPU_ID python eval_imagenet_shard.py \
                    --data_path $DATA_PATH \
                    --num_samples $PER_SHARD \
                    --num_steps $STEPS \
                    --guidance_scale $CFG \
                    --seed $((42 + SHARD_ID)) \
                    --shard_id $SHARD_ID \
                    --num_shards $NUM_SHARDS \
                    --output_dir $OUTPUT_DIR \
                    --ckpt_every 25
            " &
        SHARD_ID=$((SHARD_ID + 1))
    done
done
wait

echo "=== All 8 shards done. Computing metrics... ==="

python eval_imagenet_combine.py \
    --data_path $DATA_PATH \
    --shard_dir "$OUTPUT_DIR" \
    --num_shards $NUM_SHARDS \
    --num_steps $STEPS \
    --guidance_scale $CFG \
    --model_name $MODEL_NAME

echo "Done!"
