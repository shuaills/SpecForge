#!/bin/bash
# Master script to run all Eagle3 scaling law experiments
# This will run overnight - each experiment takes ~2-4 hours on 8 GPUs

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

echo "Starting Eagle3 scaling law experiments..."
echo "Estimated total time: 12-16 hours"
echo "All experiments will log to wandb project: eagle3-scaling-laws"
echo "Timestamp: $(date)"

# Create log directory
LOG_DIR="$ROOT_DIR/scaling_logs"
mkdir -p $LOG_DIR

# Function to run experiment with logging
run_experiment() {
    local model_size=$1
    local data_frac=$2
    local log_file="$LOG_DIR/experiment_${model_size}_${data_frac}data.log"

    echo "[$(date)] Starting experiment: $model_size with ${data_frac} data" | tee -a $log_file
    bash $SCRIPT_DIR/run_llama3_eagle3_scaling.sh $model_size $data_frac 8 2>&1 | tee -a $log_file
    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        echo "[$(date)] ✅ SUCCESS: $model_size with ${data_frac} data" | tee -a $log_file
    else
        echo "[$(date)] ❌ FAILED: $model_size with ${data_frac} data (exit code: $exit_code)" | tee -a $log_file
    fi
    echo "" | tee -a $log_file
}

# Parameter scaling experiments (fixed 1 layer, varying hidden size)
echo "🔬 Running Parameter Scaling Experiments (hidden size scaling)..."
run_experiment "tiny" "1.0"     # 512 hidden, ~1M params
run_experiment "small" "1.0"    # 1024 hidden, ~8M params
run_experiment "medium" "1.0"   # 2048 hidden, ~50M params
run_experiment "original" "1.0" # 4096 hidden, ~530M params
run_experiment "large" "1.0"    # 6144 hidden, ~1.2B params
run_experiment "pro" "1.0"      # 8192 hidden, ~2.1B params

# Layer scaling experiments (fixed 4096 hidden, varying layers)
echo "📚 Running Layer Scaling Experiments (depth scaling)..."
run_experiment "original" "1.0" # 1 layer (baseline)
run_experiment "2layer" "1.0"   # 2 layers
run_experiment "4layer" "1.0"   # 4 layers

# Data scaling experiments (fixed medium model, varying data)
echo "📊 Running Data Scaling Experiments..."
run_experiment "medium" "0.1"   # 10% data
run_experiment "medium" "0.5"   # 50% data
run_experiment "medium" "1.0"   # 100% data (already done above)

echo "🎉 All scaling law experiments completed!"
echo "Check logs in: $LOG_DIR"
echo "View results in wandb project: eagle3-scaling-laws"
echo "Finished at: $(date)"
