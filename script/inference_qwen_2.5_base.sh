#!/bin/bash

# Setup uv
uv sync
source .venv/bin/activate

mkdir log/qwen2.5_0.5_base -p

######### Run inference on qwen2.5-0.5-base model
uv run python -m \
    tiny_reasoning_model.run_evaluate \
    config/qwen2.5_0.5_base/math500_base.yaml \
    > log/qwen2.5_0.5_base/math500_base.txt 2>&1

uv run python -m \
    tiny_reasoning_model.run_evaluate \
    config/qwen2.5_0.5_base/math500_base_800.yaml \
    > log/qwen2.5_0.5_base/math500_base_800.txt 2>&1

uv run python -m \
    tiny_reasoning_model.run_evaluate \
    config/qwen2.5_0.5_base/math500_cot.yaml \
    > log/qwen2.5_0.5_base/math500_cot.txt 2>&1

uv run python -m \
    tiny_reasoning_model.run_evaluate \
    config/qwen2.5_0.5_base/math500_cot_800.yaml \
    > log/qwen2.5_0.5_base/math500_cot_800.txt 2>&1

uv run python -m \
    tiny_reasoning_model.run_evaluate \
    config/qwen2.5_0.5_base/math500_self_consistency_3.yaml \
    > log/qwen2.5_0.5_base/math500_self_consistency_3.txt 2>&1

uv run python -m \
    tiny_reasoning_model.run_evaluate \
    config/qwen2.5_0.5_base/math500_self_consistency_3_cot800.yaml \
    > log/qwen2.5_0.5_base/math500_self_consistency_3_cot800.txt 2>&1

uv run python -m \
    tiny_reasoning_model.run_evaluate \
    config/qwen2.5_0.5_base/math500_self_consistency_5.yaml \
    > log/qwen2.5_0.5_base/math500_self_consistency_5.txt 2>&1

uv run python -m \
    tiny_reasoning_model.run_evaluate \
    config/qwen2.5_0.5_base/math500_self_consistency_5_cot800.yaml \
    > log/qwen2.5_0.5_base/math500_self_consistency_5_cot800.txt 2>&1
