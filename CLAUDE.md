# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpecForge is a framework for training Eagle3 speculative decoding models for the SGLang serving framework. It supports two training modes:
- **Online training**: Freezes target model and trains draft model simultaneously, requires more GPU memory
- **Offline training**: Pre-generates hidden states from target model, uses less GPU memory but requires significant disk space (12TB+ for full datasets)

## Installation and Setup

```bash
# Install the package
pip install -v .

# Install SGLang for benchmarking
uv pip install "sglang[all]>=0.4.9.post2"
```

## Development Commands

### Data Preparation
```bash
# Prepare online training datasets
python scripts/prepare_data.py --dataset ultrachat
python scripts/prepare_data.py --dataset sharegpt

# Generate hidden states for offline training
torchrun --nproc_per_node=8 scripts/prepare_hidden_states.py \
    --model-path <target-model-path> \
    --enable-aux-hidden-states \
    --data-path <jsonl-file-path> \
    --chat-template llama3 \
    --max-length 2048 \
    --tp-size 8 \
    --batch-size 4 \
    --mem-frac=0.75 \
    --num-samples 1000
```

### Training
```bash
# Online training examples
bash ./examples/run_llama3_eagle3_online.sh
bash ./examples/run_llama4_eagle3_online.sh
bash ./examples/run_qwen3_dense_eagle3_online.sh
bash ./examples/run_qwen3_moe_eagle3_online.sh
bash ./examples/run_qwq_eagle3_online.sh

# Offline training
bash ./examples/run_llama3_eagle3_offline.sh

# Custom training command structure
torchrun --standalone --nproc_per_node 8 scripts/train_eagle3_online.py \
    --target-model-path <target-model> \
    --draft-model-config ./configs/<model>-eagle3.json \
    --train-data-path <dataset>.jsonl \
    --output-dir <output-path> \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template <template> \
    --cache-dir ./cache
```

### Testing
```bash
# Run tests (no specific test runner configured)
python -m pytest tests/
```

### Benchmarking with SGLang
```bash
# Start SGLang server
python3 -m sglang.launch_server \
    --model <target-model-path> \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path <draft-model-path> \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.75 \
    --cuda-graph-max-bs 2 \
    --tp 8 \
    --context-length 8192 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16

# Run benchmarks
cd benchmarks/
python run_gsm8k.py
python run_math500.py
python run_mtbench.py
python run_humaneval.py
```

## Architecture Overview

### Core Models
- `OnlineEagle3Model`: Implements online training with TTT (test-time training) technique
- `OfflineEagle3Model`: Implements offline training using pre-computed hidden states
- `QwenVLOnlineEagle3Model`: Vision-language model variant for Qwen2.5-VL

### Key Components
- **specforge/core/eagle3.py**: Main Eagle3 model implementations with TTT logic
- **specforge/modeling/auto.py**: Auto model loading for target and draft models
- **specforge/modeling/draft/**: Draft model implementations (currently supports Llama architecture)
- **specforge/modeling/target/**: Target model implementations with tensor parallelism support
- **specforge/data/**: Dataset preparation and processing utilities
- **configs/**: Model configuration files for different architectures

### Training Process
1. Extract hidden states from 3 auxiliary layers (layer 1, middle layer, layer n-4)
2. Concatenate and project to target hidden size
3. Run TTT for specified length (default 7 steps)
4. Train draft model to predict target model outputs

### Dataset Format
Training data should be in JSONL format:
```json
{
    "id": "xxxx",
    "conversations": [
        {
            "role": "user | assistant",
            "content": "The message content"
        }
    ]
}
```

### Chat Templates
Register new chat templates in `specforge/data/template.py` using the `TEMPLATE_REGISTRY`.

### Adding New Models
- For target models: Implement in `specforge/modeling/target/` inheriting `DistributedTargetModel`
- For draft models: Implement in `specforge/modeling/draft/` inheriting `Eagle3DraftModel`
- Register both in `specforge/modeling/auto.py`
