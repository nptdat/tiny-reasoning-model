# tiny-reasoning-model
A lightweight implementation of inference-time scaling and training-time scaling methods, designed for experimentation and deeper understanding.

# Base models
- Qwen/Qwen2.5-0.5B: https://huggingface.co/Qwen/Qwen2.5-0.5B

# Assumption on evaluation
We assume the followings to simplify the evaluation
- Limit to math questions which have the final result in the following forms:
    - Exact number, exact text
    - Fomula (polynomial...)
- Judge on the final result only. Not care about the explanation of the answer. 


# Setup

```
uv sync
source .venv/bin/activate
```

# To run test

```
uv run python -m pytest
```

# To run inference
- With uv

```
uv run python -m tiny_reasoning_model.run_evaluate config/qwen2.5_0.5_base/math500_base.yaml
```

- With shellscript

```
sh script/inference_qwen_2.5_base.sh
```


# To fine-tune a model with grpo
- With uv

```
uv run python -m tiny_reasoning_model.run_train_grpo config/qwen2.5_0.5_base/train_grpo_no_kl_1000steps.yaml
```

- With shellscript

```
sh script/train_qwen_2.5_base.sh
```


# To run the whole pipeline
- NOTE: This will take very long time (days). It runs the following sub-pipelines:
    - Inference on base model
    - Train the base model with GRPO-no-kl and GRPO-with-kl
    - Inference on fine-tuned models (multiple checkpoints for GRPO-no-kl and GRPO-with-kl)
- This script is to illustrate the whole pipeline. You should run each sub-pipeline separately (recommended).

```
sh script/evaluate_all.sh
```
