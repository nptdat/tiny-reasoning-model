#!/bin/bash

# Setup uv
uv sync
source .venv/bin/activate

mkdir log/qwen2.5_0.5_base -p

# Run train on qwen2.5-0.5-base model
uv run python -m \
    tiny_reasoning_model.run_train_grpo \
    config/qwen2.5_0.5_base/train_grpo_no_kl_1000steps.yaml \
    > log/qwen2.5_0.5_base/train_grpo_no_kl_1000steps.txt 2>&1

uv run python -m \
    tiny_reasoning_model.run_train_grpo \
    config/qwen2.5_0.5_base/train_grpo_kl_1000steps.yaml \
    > log/qwen2.5_0.5_base/train_grpo_kl_1000steps.txt 2>&1
