#!/bin/bash

# Setup uv
uv sync
source .venv/bin/activate

mkdir log/qwen2.5_0.5_base -p

######### Run inference on fine-tuned qwen2.5-0.5-base model WITH KL
uv run python -m \
    tiny_reasoning_model.run_evaluate \
    config/qwen2.5_0.5_base/math500_grpo_kl_50steps_base.yaml \
    > log/qwen2.5_0.5_base/math500_grpo_kl_50steps_base.txt 2>&1

uv run python -m \
    tiny_reasoning_model.run_evaluate \
    config/qwen2.5_0.5_base/math500_grpo_kl_100steps_base.yaml \
    > log/qwen2.5_0.5_base/math500_grpo_kl_100steps_base.txt 2>&1

uv run python -m \
    tiny_reasoning_model.run_evaluate \
    config/qwen2.5_0.5_base/math500_grpo_kl_150steps_base.yaml \
    > log/qwen2.5_0.5_base/math500_grpo_kl_150steps_base.txt 2>&1

uv run python -m \
    tiny_reasoning_model.run_evaluate \
    config/qwen2.5_0.5_base/math500_grpo_kl_200steps_base.yaml \
    > log/qwen2.5_0.5_base/math500_grpo_kl_200steps_base.txt 2>&1

uv run python -m \
    tiny_reasoning_model.run_evaluate \
    config/qwen2.5_0.5_base/math500_grpo_kl_400steps_base.yaml \
    > log/qwen2.5_0.5_base/math500_grpo_kl_400steps_base.txt 2>&1

uv run python -m \
    tiny_reasoning_model.run_evaluate \
    config/qwen2.5_0.5_base/math500_grpo_kl_600steps_base.yaml \
    > log/qwen2.5_0.5_base/math500_grpo_kl_600steps_base.txt 2>&1


uv run python -m \
    tiny_reasoning_model.run_evaluate \
    config/qwen2.5_0.5_base/math500_grpo_kl_50steps_cot.yaml \
    > log/qwen2.5_0.5_base/math500_grpo_kl_50steps_cot.txt 2>&1

uv run python -m \
    tiny_reasoning_model.run_evaluate \
    config/qwen2.5_0.5_base/math500_grpo_kl_100steps_cot.yaml \
    > log/qwen2.5_0.5_base/math500_grpo_kl_100steps_cot.txt 2>&1

uv run python -m \
    tiny_reasoning_model.run_evaluate \
    config/qwen2.5_0.5_base/math500_grpo_kl_150steps_cot.yaml \
    > log/qwen2.5_0.5_base/math500_grpo_kl_150steps_cot.txt 2>&1

uv run python -m \
    tiny_reasoning_model.run_evaluate \
    config/qwen2.5_0.5_base/math500_grpo_kl_200steps_cot.yaml \
    > log/qwen2.5_0.5_base/math500_grpo_kl_200steps_cot.txt 2>&1

uv run python -m \
    tiny_reasoning_model.run_evaluate \
    config/qwen2.5_0.5_base/math500_grpo_kl_400steps_cot.yaml \
    > log/qwen2.5_0.5_base/math500_grpo_kl_400steps_cot.txt 2>&1

uv run python -m \
    tiny_reasoning_model.run_evaluate \
    config/qwen2.5_0.5_base/math500_grpo_kl_600steps_cot.yaml \
    > log/qwen2.5_0.5_base/math500_grpo_kl_600steps_cot.txt 2>&1
