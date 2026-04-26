import yaml
from typing import Self
from enum import StrEnum

from pathlib import Path
from pydantic import BaseModel, model_validator


class SamplingMethod(StrEnum):
    GREEDY = "greedy"
    TEMPERATURE = "temperature"
    TOP_K = "top_k"
    TOP_P = "top_p"


class PromptingMode(StrEnum):
    BASE = "base-prompting"
    COT = "cot-prompting"


class InferenceTimeScaling(StrEnum):
    NO = "no"
    SELF_CONSISTENCY = "self-consistency"
    SELF_REFINEMENT = "self-refinement"


class ConfigCoTPrompting(BaseModel):
    template: str


class ConfigSelfConsistency(BaseModel):
    num_samples: int


class ConfigSelfRefinement(BaseModel):
    num_iterations: int
    critique_prompt: str
    refine_prompt: str


class ConfigGenerationSampling(BaseModel):
    # If this is None -> greedy sampling
    # If both of top_k and top_k are None -> softmax sampling wihout filter
    # If top_k is not None -> top-k sampling
    # If top_p is not None -> top-p sampling
    sampling_method_name: SamplingMethod = SamplingMethod.TEMPERATURE
    # sampling_method_name: str
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None

    @model_validator(mode="after")
    def validate_top_p_k(self) -> Self:
        if self.sampling_method_name == SamplingMethod.GREEDY:
            if (
                self.temperature is not None
                or self.top_k is not None
                or self.top_p is not None
            ):
                raise ValueError(
                    f"For greedy sampling, temperature, top_k, and top_p must be None. Got temperature={self.temperature}, top_k={self.top_k}, top_p={self.top_p}"
                )
        elif self.sampling_method_name == SamplingMethod.TEMPERATURE:
            if (
                self.temperature is None
                or self.top_k is not None
                or self.top_p is not None
            ):
                raise ValueError(
                    f"For temperature sampling, temperature must be not None, and top_k, and top_p must be None. Got temperature={self.temperature}, top_k={self.top_k}, top_p={self.top_p}"
                )
        elif self.sampling_method_name == SamplingMethod.TOP_P:
            if self.temperature is None or self.top_k is not None or self.top_p is None:
                raise ValueError(
                    f"For top_p filtering, temperature and top_p must be not None, and top_k must be None. Got temperature={self.temperature}, top_k={self.top_k}, top_p={self.top_p}"
                )
        elif self.sampling_method_name == SamplingMethod.TOP_K:
            if self.temperature is None or self.top_k is None or self.top_p is not None:
                raise ValueError(
                    f"For top_p filtering, temperature and top_k must be not None, and top_p must be None. Got temperature={self.temperature}, top_k={self.top_k}, top_p={self.top_p}"
                )
        return self


class Config(BaseModel):
    seed: int = 42
    generation_sampling: ConfigGenerationSampling
    model_path: str
    dataset_path: str
    result_path: str | None = None
    max_new_tokens: int = 500
    eos_token: int | None = None
    max_samples: int | None = None

    prompting_mode: PromptingMode = PromptingMode.BASE
    inference_time_scaling: InferenceTimeScaling = InferenceTimeScaling.NO

    cot_prompting: ConfigCoTPrompting | None = None
    self_consistency: ConfigSelfConsistency | None = None
    self_refinement: ConfigSelfRefinement | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    @model_validator(mode="after")
    def validate_top_p_k(self) -> Self:
        if self.prompting_mode == PromptingMode.COT:
            if self.cot_prompting is None:
                raise ValueError(
                    f"For cot prompting, cot_prompting must be not None. Got cot_prompting={self.cot_prompting}"
                )
        elif self.prompting_mode == PromptingMode.BASE:
            if self.cot_prompting is not None:
                raise ValueError(
                    f"For base prompting, cot_prompting must be None. Got cot_prompting={self.cot_prompting}"
                )

        if self.inference_time_scaling == InferenceTimeScaling.NO:
            if self.self_consistency is not None or self.self_refinement is not None:
                raise ValueError(
                    f"For no inference time scaling, self_consistency and self_refinement must be None. Got self_consistency={self.self_consistency}, self_refinement={self.self_refinement}"
                )
        elif self.inference_time_scaling == InferenceTimeScaling.SELF_CONSISTENCY:
            if self.self_consistency is None:
                raise ValueError(
                    f"For self-consistency, self_consistency must be not None. Got self_consistency={self.self_consistency}"
                )
            if self.self_refinement is not None:
                raise ValueError(
                    f"For self-consistency, self_refinement must be None. Got self_refinement={self.self_refinement}"
                )
            if self.generation_sampling.sampling_method_name == SamplingMethod.GREEDY:
                raise ValueError(
                    f"For self-consistency, sampling method must be not `greedy`. Got sampling_method_name={self.generation_sampling.sampling_method_name}"
                )
        elif self.inference_time_scaling == InferenceTimeScaling.SELF_REFINEMENT:
            if self.self_refinement is None:
                raise ValueError(
                    f"For self-refinement, self_refinement must be not None. Got self_refinement={self.self_refinement}"
                )
            if self.self_consistency is not None:
                raise ValueError(
                    f"For self-refinement, self_consistency must be None. Got self_consistency={self.self_consistency}"
                )

        return self


class ConfigGRPO(BaseModel):
    """
    Definitions:
        - Step: sample
        - Batch: number of outputs to optimize model for once.
    There may be multiple batches per sample (`batches_per_step`)
    Update model every batch of G outputs.
    Update the model_old for every sample.
    Update reference model every `update_reference_every_steps` steps.
    """

    seed: int = 42
    model_path: str
    generation_sampling: ConfigGenerationSampling
    max_new_tokens: int = 500
    eos_token: int | None = None
    prompting_mode: PromptingMode = PromptingMode.BASE
    cot_prompting: ConfigCoTPrompting | None = None

    # number of outputs per sample to optimize model for once.
    num_outputs: int = 4
    # Number of samples to update the model_reference for once
    update_reference_every_steps: int = 16
    # Number of batches to update the mode_old for once
    batches_per_step: int = 2

    learning_rate: float = 1e-5
    clip_grad_norm: float = 1.0

    clip_epsilon: float = 10.0
    # weight of the KL loss
    beta: float = 0.001

    # TODO: use validation as in Config

    # Currently always use rasbt/math_full_minus_math500 from utils.load_math_data
    # train_dataset_path: str
    train_max_steps: int | None = None
    eval_dataset_path: str | None = None
    eval_max_samples: int = 50
    eval_every: int = 100
    checkpoint_every_steps: int = 10

    output_dir: str = "model"
    log_dir: str = "log/runs"  # TensorBoard log dir

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConfigGRPO":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)
