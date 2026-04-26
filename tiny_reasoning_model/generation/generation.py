from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from tiny_reasoning_model.schema.config import (
    ConfigGenerationSampling,
    SamplingMethod as Sampling,
)


def _greedy_sampling(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    token_prob = probs[0][int(next_token.item())]
    return next_token, token_prob


def _temperature_sampling(
    logits: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature != 1.0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    token_prob = probs[int(next_token.item())]
    return next_token, token_prob


def _top_k_sampling(
    logits: torch.Tensor, temperature: float, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature != 1.0:
        logits = logits / temperature
    probs = torch.softmax(
        logits, dim=-1
    ).flatten()  # disable batch dimension. Assume batch size is 1
    indices = probs.argsort(dim=-1, descending=True)[:top_k]
    new_probs = torch.zeros_like(probs)
    new_probs[indices] = probs[indices]
    new_probs = new_probs / new_probs.sum()
    next_token = torch.multinomial(new_probs, num_samples=1).unsqueeze(dim=-1)
    token_prob = probs[int(next_token.item())]
    return next_token, token_prob


def _top_p_sampling(
    logits: torch.Tensor, temperature: float, top_p: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature != 1.0:
        logits = logits / temperature
    probs = torch.softmax(
        logits, dim=-1
    ).flatten()  # disable batch dimension. Assume batch size is 1
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs > top_p
    top_k = mask.shape[0] - mask.sum()

    if top_k == 0:
        top_k += 1
    indices = sorted_indices[:top_k]

    if len(indices) == 1:
        next_token = indices.unsqueeze(dim=-1)
    else:
        new_probs = torch.zeros_like(probs)
        new_probs[indices] = probs[indices]
        new_probs = new_probs / new_probs.sum()
        next_token = torch.multinomial(new_probs, num_samples=1).unsqueeze(dim=-1)

    token_prob = probs[int(next_token.item())]
    return next_token, token_prob


# @torch.inference_mode()
def generate(
    model: PreTrainedModel,
    token_ids: torch.Tensor,
    generation_sampling: ConfigGenerationSampling,
    max_new_tokens: int,
    eos_token: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    method_name = generation_sampling.sampling_method_name
    temperature = generation_sampling.temperature
    top_k = generation_sampling.top_k
    top_p = generation_sampling.top_p

    token_probs = []
    for _ in range(max_new_tokens):
        logits = model(token_ids).logits[:, -1]

        if method_name == Sampling.GREEDY:
            next_token, token_prob = _greedy_sampling(logits)
        elif method_name == Sampling.TEMPERATURE:
            next_token, token_prob = _temperature_sampling(logits, temperature)
        elif method_name == Sampling.TOP_K:
            next_token, token_prob = _top_k_sampling(logits, temperature, top_k)
        elif method_name == Sampling.TOP_P:
            next_token, token_prob = _top_p_sampling(logits, temperature, top_p)
        else:
            raise ValueError(f"Sampling method `{method_name}` is unidentified!")

        if eos_token is not None:
            if next_token.item() == eos_token:
                break
        token_ids = torch.cat([token_ids, next_token], dim=-1)
        token_probs.append(token_prob)

    log_probs = torch.log(torch.tensor(token_probs))

    return token_ids, log_probs


def generate_solution(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    generation_sampling: ConfigGenerationSampling,
    max_new_tokens: int = 500,
    eos_token: int | None = None,
) -> tuple[str, float]:
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    prompt_len = len(model_inputs["input_ids"][0])
    out_tokens, log_probs = generate(
        model,
        model_inputs["input_ids"],
        generation_sampling,
        max_new_tokens,
        eos_token=eos_token,
    )

    log_probs = log_probs[prompt_len:]
    avg_log_prob = log_probs.mean().item()

    decoded_text = tokenizer.decode(out_tokens[0], skip_special_tokens=True)
    assert isinstance(decoded_text, str)

    return decoded_text, avg_log_prob


# @torch.inference_mode()
def compute_log_probs(
    model: PreTrainedModel,
    token_ids: torch.Tensor,
    prompt_len: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sequence-level log_probs
    """
    logits = model(token_ids).logits

    if temperature != 1.0:
        logits = logits / temperature
    log_probs = torch.log_softmax(logits, dim=-1)[0]
    selected = log_probs[:-1].gather(1, token_ids[0][1:].unsqueeze(-1)).squeeze(-1)
    sum_log_probs = torch.sum(selected[prompt_len - 1 :]) # scalar
    return sum_log_probs
