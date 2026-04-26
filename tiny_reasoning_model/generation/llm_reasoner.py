from collections import Counter, defaultdict

from jinja2 import Template
from transformers import PreTrainedModel, PreTrainedTokenizer

from .generation import generate_solution
from tiny_reasoning_model.schema.config import (
    ConfigCoTPrompting,
    InferenceTimeScaling,
    PromptingMode,
)
from tiny_reasoning_model.expression import extract_answer
from tiny_reasoning_model.schema import Config


# build prompt from static template
def build_prompt(problem: str, prompt_template: str) -> str:
    template = Template(prompt_template)
    return template.render(problem=problem)


# def build_raw_prompt(problem: str, config: Config) -> str:
#     if config.prompting_mode == PromptingMode.BASE:
#         prompt = problem
#     elif config.prompting_mode == PromptingMode.COT:
#         assert config.cot_prompting is not None, "cot_prompting must be configured"
#         prompt = build_prompt(problem, config.cot_prompting.template)
#     else:
#         raise ValueError("mode must be either 'base-prompting' or 'cot-prompting'")
#     return prompt


def build_raw_prompt(
    problem: str,
    prompting_mode: PromptingMode,
    cot_prompting: ConfigCoTPrompting | None = None,
) -> str:
    if prompting_mode == PromptingMode.BASE:
        prompt = problem
    elif prompting_mode == PromptingMode.COT:
        assert cot_prompting is not None, "cot_prompting must be configured"
        prompt = build_prompt(problem, cot_prompting.template)
    else:
        raise ValueError("mode must be either 'base-prompting' or 'cot-prompting'")
    return prompt


class SelfConsistencyReasoner:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    config: Config

    def __init__(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: Config
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def solve(self, problem: str) -> str:
        cfg = self.config
        assert cfg.self_consistency is not None, "self_consistency must be configured"
        num_samples = cfg.self_consistency.num_samples
        prompt = build_raw_prompt(problem, cfg.prompting_mode, cfg.cot_prompting)
        counter: Counter[str] = Counter()
        solutions: defaultdict[str, list[str]] = defaultdict(list)
        best_solution: str | None = None
        for _ in range(num_samples):
            solution, _ = generate_solution(
                self.model,
                self.tokenizer,
                prompt,
                cfg.generation_sampling,
                max_new_tokens=cfg.max_new_tokens,
                eos_token=cfg.eos_token,
            )
            answer = extract_answer(solution)
            if answer:
                # Assuming extract_answer returns a string when truthy
                answer_str = str(answer)
                solutions[answer_str].append(solution)
                counter[answer_str] += 1
            else:
                if best_solution is None:
                    best_solution = solution

        if len(counter) > 0:
            best_answer = counter.most_common(1)[0][0]
            best_solution = solutions[best_answer][0]

        if best_solution is None:
            raise ValueError("Failed to generate any solution.")

        return best_solution


class SelfRefinementReasoner:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    config: Config

    def __init__(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: Config
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def solve(self, problem: str) -> str:
        cfg = self.config.self_refinement
        assert cfg is not None, "self_refinement must be configured"

        best_solution: str
        best_score = float("-inf")
        for _ in range(cfg.num_iterations):
            # STEP1: generate draft solution
            basic_prompt = build_raw_prompt(
                problem, self.config.prompting_mode, self.config.cot_prompting
            )
            draft_solution, draft_score = generate_solution(
                self.model,
                self.tokenizer,
                basic_prompt,
                self.config.generation_sampling,
                max_new_tokens=self.config.max_new_tokens,
                eos_token=self.config.eos_token,
            )
            best_solution = draft_solution[len(basic_prompt) :].strip()
            # best_solution = draft_solution
            best_score = draft_score

            # STEP2: generate critique
            critique_prompt = Template(cfg.critique_prompt).render(
                problem=problem, draft_solution=draft_solution
            )
            critique, _ = generate_solution(
                self.model,
                self.tokenizer,
                critique_prompt,
                self.config.generation_sampling,
                max_new_tokens=self.config.max_new_tokens,
                eos_token=self.config.eos_token,
            )
            critique = critique[len(critique_prompt) :].strip()

            # STEP3: generate refined solution
            refine_prompt = Template(cfg.refine_prompt).render(
                problem=problem, draft_solution=draft_solution, critique=critique
            )
            refined_solution, refined_score = generate_solution(
                self.model,
                self.tokenizer,
                refine_prompt,
                self.config.generation_sampling,
                max_new_tokens=self.config.max_new_tokens,
                eos_token=self.config.eos_token,
            )
            if refined_score > best_score:
                best_solution = refined_solution[len(refine_prompt) :].strip()
                # best_solution = refined_solution
                best_score = refined_score

        return best_solution


class LLMReasoner:
    """
    LLM-based reasoner for math problems.
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    config: Config

    def __init__(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: Config
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def solve(self, problem: str) -> str:
        cfg = self.config
        solution: str
        if cfg.inference_time_scaling == InferenceTimeScaling.NO:
            prompt = build_raw_prompt(problem, cfg.prompting_mode, cfg.cot_prompting)
            solution, _ = generate_solution(
                self.model,
                self.tokenizer,
                prompt,
                cfg.generation_sampling,
                max_new_tokens=cfg.max_new_tokens,
                eos_token=cfg.eos_token,
            )
        elif cfg.inference_time_scaling == InferenceTimeScaling.SELF_CONSISTENCY:
            self_consistency_reasoner = SelfConsistencyReasoner(
                self.model, self.tokenizer, cfg
            )
            solution = self_consistency_reasoner.solve(problem)
        elif cfg.inference_time_scaling == InferenceTimeScaling.SELF_REFINEMENT:
            self_refinement_reasoner = SelfRefinementReasoner(
                self.model, self.tokenizer, cfg
            )
            solution = self_refinement_reasoner.solve(problem)
        else:
            raise ValueError(
                f"Unsupported inference time scaling: {cfg.inference_time_scaling}"
            )

        return solution
