"""
References:
[1] https://arxiv.org/pdf/2402.03300 (DeepSeekMath)
[2] https://arxiv.org/pdf/2501.12948 (DeepSeek-R1)
"""

import copy
import logging

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from tiny_reasoning_model.schema.config import ConfigGRPO
from tiny_reasoning_model.utils import get_device, load_math_data, set_seed
from tiny_reasoning_model.generation import compute_log_probs, generate, build_raw_prompt
from tiny_reasoning_model.evaluator import Grader


logger = logging.getLogger(__name__)


def get_rewards(solution: str, correct_answer: str) -> float:
    correct = Grader.grade(solution, correct_answer)
    return correct


def set_require_grad(model: PreTrainedModel, value: bool) -> None:
    for p in model.parameters():
        p.requires_grad = value


def compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    return advantages


class GRPO:
    def __init__(self, config: ConfigGRPO):
        self.config = config
        self.device = get_device()

    def train(self):
        cfg = self.config

        set_seed(cfg.seed)

        # currently, use a fixed dataset (math_full_minus_math500) to train
        train_ds = load_math_data()

        G = cfg.num_outputs
        max_steps = cfg.train_max_steps or len(train_ds)

        writer = SummaryWriter(log_dir=cfg.log_dir)

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path)
        model = model.to(self.device)

        if cfg.beta > 0:
            model_ref = copy.deepcopy(model)
            set_require_grad(model_ref, False)
            model_ref.eval()

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        model.train()
        optimizer.zero_grad()

        for step in range(max_steps):
            logger.info(f"------- {step=} -------")
            sample = train_ds[step]
            model_old = copy.deepcopy(model)
            set_require_grad(model_old, False)
            model_old.eval()
            prompt = build_raw_prompt(
                sample["problem"], cfg.prompting_mode, cfg.cot_prompting
            )
            for b in range(cfg.batches_per_step):
                logger.info(f"{b=}")

                # generate G outputs
                rewards = []
                new_log_probs = []
                old_log_probs = []
                ref_log_probs = []
                gen_solutions = []
                for g in range(G):
                    logger.info(f"\t{g=}")
                    model_inputs = tokenizer([prompt], return_tensors="pt").to(
                        model.device
                    )
                    input_ids = model_inputs["input_ids"]
                    prompt_len = input_ids.shape[1]
                    with torch.no_grad():
                        token_ids, _ = generate(
                            model_old,
                            input_ids,
                            cfg.generation_sampling,
                            cfg.max_new_tokens,
                            eos_token=cfg.eos_token,
                        )

                        old_log_probs.append(
                            compute_log_probs(model_old, token_ids, prompt_len)
                        )
                        if cfg.beta > 0:
                            ref_log_probs.append(
                                compute_log_probs(model_ref, token_ids, prompt_len)
                            )

                    # Also recompute the log_probs for new model with temperature = 1
                    new_log_probs.append(
                        compute_log_probs(model, token_ids, prompt_len)
                    )

                    solution_text = tokenizer.decode(
                        token_ids[0], skip_special_tokens=True
                    )
                    gen_solutions.append(solution_text)
                    reward = float(get_rewards(solution_text, sample["answer"]))
                    rewards.append(reward)

                # Compute advantages A
                A = compute_advantages(torch.tensor(rewards, device=self.device))
                A.detach()

                logger.info(f"{rewards=}")
                logger.info(f"{A=}")

                # stack a list of scalar tensors to a 1-d tensor
                # breakpoint()
                new_log_probs = torch.stack(new_log_probs)
                old_log_probs = torch.stack(old_log_probs).detach()

                # Compute loss
                log_new_old_ratio = new_log_probs - old_log_probs
                new_old_ratio = torch.exp(log_new_old_ratio)
                logger.info(f"{log_new_old_ratio=}")
                logger.info(f"{new_old_ratio=}")
                clip_new_old_ratio = torch.clip(
                    new_old_ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon
                )

                # From [1] & [2], maximize the object -> minimize the negative object
                pg_loss = -torch.min(new_old_ratio * A, clip_new_old_ratio * A).mean()

                if cfg.beta > 0:
                    ref_log_probs = torch.stack(ref_log_probs).detach()

                    log_new_ref_ratio = new_log_probs - ref_log_probs
                    new_ref_ratio = torch.exp(log_new_ref_ratio)
                    logger.info(f"{log_new_ref_ratio=}")
                    logger.info(f"{new_ref_ratio=}")
                    # minimize the negative object -> KL[P_ref || P_new]
                    kl_loss = (new_ref_ratio - log_new_ref_ratio - 1.0).mean()

                    loss = pg_loss - cfg.beta * kl_loss
                else:
                    loss = pg_loss

                # optimize the loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                optimizer.step()

                logger.info(f"{pg_loss.item()=:.8f}")
                if cfg.beta > 0:
                    logger.info(f"{kl_loss.item()=:.8f}")
                logger.info(f"{loss.item()=:.8f}")

                global_step = step * cfg.batches_per_step + b

                avg_gen_lens = (
                    np.array([len(gen_solution) for gen_solution in gen_solutions])
                    .mean()
                    .item()
                )
                writer.add_scalar("generation/avg_gen_len", avg_gen_lens, global_step)
                writer.add_scalar("generation/reward_max", max(rewards), global_step)
                writer.add_scalar(
                    "generation/reward_mean",
                    torch.tensor(rewards).mean().item(),
                    global_step,
                )
                writer.add_scalar(
                    "generation/reward_std",
                    torch.tensor(rewards).std().item(),
                    global_step,
                )
                # writer.add_scalar("generation/advantage_mean", A.mean().item(), global_step) # always 0 -> ignore
                writer.add_scalar(
                    "generation/advantage_std", A.std().item(), global_step
                )
                writer.add_scalar("loss/total", loss.item(), global_step)
                writer.add_scalar("loss/policy_gradient", pg_loss.item(), global_step)
                if cfg.beta > 0:
                    writer.add_scalar("loss/kl", kl_loss.item(), global_step)

                writer.add_scalar(
                    "ratio/log_new_old_ratio_mean",
                    log_new_old_ratio.mean().item(),
                    global_step,
                )
                writer.add_scalar(
                    "ratio/new_old_ratio_mean", new_old_ratio.mean().item(), global_step
                )
                if cfg.beta > 0:
                    writer.add_scalar(
                        "ratio/log_new_ref_ratio_mean",
                        log_new_ref_ratio.mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "ratio/new_ref_ratio_mean",
                        new_ref_ratio.mean().item(),
                        global_step,
                    )

            if (step + 1) % cfg.checkpoint_every_steps == 0:
                model.save_pretrained(f"{cfg.output_dir}/step_{step + 1}")
                tokenizer.save_pretrained(f"{cfg.output_dir}/step_{step + 1}")

            if (step + 1) % cfg.update_reference_every_steps == 0 and cfg.beta > 0:
                model_ref = copy.deepcopy(model)
                set_require_grad(model_ref, False)
                model_ref.eval()

            # TODO: Evaluation

        writer.close()
