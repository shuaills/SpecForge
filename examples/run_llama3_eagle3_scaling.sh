#!/bin/bash
# Eagle3 Scaling Law Experiments
# Usage: bash run_llama3_eagle3_scaling.sh [model_size] [data_fraction] [num_gpus]
# model_size: tiny, small, medium, original (default: tiny)
# data_fraction: fraction of dataset to use 0.1-1.0 (default: 0.1)
# num_gpus: number of GPUs to use (default: 1)

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# Parse arguments
MODEL_SIZE=${1:-tiny}
DATA_FRACTION=${2:-1.0}
NUM_GPUS=${3:-8}

# Set config based on model size
case $MODEL_SIZE in
    "tiny")
        CONFIG_FILE="$ROOT_DIR/configs/llama3-8B-eagle3-tiny.json"
        EXP_NAME="eagle3-tiny-${DATA_FRACTION}data"
        ;;
    "small")
        CONFIG_FILE="$ROOT_DIR/configs/llama3-8B-eagle3-small.json"
        EXP_NAME="eagle3-small-${DATA_FRACTION}data"
        ;;
    "medium")
        CONFIG_FILE="$ROOT_DIR/configs/llama3-8B-eagle3-medium.json"
        EXP_NAME="eagle3-medium-${DATA_FRACTION}data"
        ;;
    "large")
        CONFIG_FILE="$ROOT_DIR/configs/llama3-8B-eagle3-large.json"
        EXP_NAME="eagle3-large-${DATA_FRACTION}data"
        ;;
    "pro")
        CONFIG_FILE="$ROOT_DIR/configs/llama3-8B-eagle3-pro.json"
        EXP_NAME="eagle3-pro-${DATA_FRACTION}data"
        ;;
    "original")
        CONFIG_FILE="$ROOT_DIR/configs/llama3-8B-eagle3.json"
        EXP_NAME="eagle3-original-${DATA_FRACTION}data"
        ;;
    "2layer")
        CONFIG_FILE="$ROOT_DIR/configs/llama3-8B-eagle3-2layer.json"
        EXP_NAME="eagle3-2layer-${DATA_FRACTION}data"
        ;;
    "4layer")
        CONFIG_FILE="$ROOT_DIR/configs/llama3-8B-eagle3-4layer.json"
        EXP_NAME="eagle3-4layer-${DATA_FRACTION}data"
        ;;
    *)
        echo "Invalid model size. Use: tiny, small, medium, large, pro, original, 2layer, 4layer"
        exit 1
        ;;
esac

echo "Running Eagle3 scaling experiment:"
echo "  Model size: $MODEL_SIZE"
echo "  Data fraction: $DATA_FRACTION"
echo "  GPUs: $NUM_GPUS"
echo "  Config: $CONFIG_FILE"
echo "  Experiment name: $EXP_NAME"

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --draft-model-config $CONFIG_FILE \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt.jsonl \
    --output-dir $ROOT_DIR/outputs/$EXP_NAME \
    --num-epochs 2 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend flex_attention \
    --data-fraction $DATA_FRACTION \
    --report-to wandb \
    --wandb-project eagle3-scaling-laws \
    --wandb-name $EXP_NAME \
    --log-flops
