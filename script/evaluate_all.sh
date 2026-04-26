#!/bin/bash

sh script/inference_qwen_2.5_base.sh

sh script/train_qwen_2.5_base.sh

sh script/inference_qwen_2.5_finetune_grpo_no_kl.sh
sh script/inference_qwen_2.5_finetune_grpo_with_kl.sh
