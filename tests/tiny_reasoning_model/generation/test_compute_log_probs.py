import math
import pytest
import torch
from unittest.mock import MagicMock

from tiny_reasoning_model.generation.generation import compute_log_probs


def _make_model(logits: torch.Tensor) -> MagicMock:
    output = MagicMock()
    output.logits = logits
    model = MagicMock()
    model.return_value = output
    return model


def _uniform_logits(seq_len: int, vocab_size: int) -> torch.Tensor:
    """Batch-size-1 logits that produce a uniform distribution over vocab."""
    return torch.zeros(1, seq_len, vocab_size)


def _run(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    prompt_len: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    model = _make_model(logits)
    return compute_log_probs(model, token_ids, prompt_len, temperature)


# ---------------------------------------------------------------------------
# Parametrize groups
# ---------------------------------------------------------------------------

uniform_cases = pytest.mark.parametrize(
    "vocab,seq_len,prompt_len",
    [
        pytest.param(8, 6, 2, id="vocab=8,seq=6,prompt=2"),
        pytest.param(4, 4, 1, id="prompt_len=1 all tokens are completion"),
        pytest.param(4, 5, 4, id="prompt_len=seq-1 single completion token"),
    ],
)

temperature_cases = pytest.mark.parametrize(
    "temperature,token_id,expected_higher",
    [
        pytest.param(
            0.1,
            0,
            True,
            id="low-temp peaks distribution -> peak token gets higher log-prob",
        ),
        pytest.param(
            5.0,
            0,
            False,
            id="high-temp flattens distribution -> peak token gets lower log-prob",
        ),
    ],
)

determinism_cases = pytest.mark.parametrize(
    "vocab,seq_len,prompt_len",
    [
        pytest.param(12, 8, 3, id="random logits deterministic"),
        pytest.param(5, 5, 2, id="uniform logits deterministic"),
    ],
)

model_call_cases = pytest.mark.parametrize(
    "vocab,seq_len,prompt_len",
    [
        pytest.param(10, 5, 2, id="model called once"),
    ],
)


# ---------------------------------------------------------------------------
# Tests: test code requires some processing to compute the correct outputs -> a bit complicated
# ---------------------------------------------------------------------------


@uniform_cases
def test_uniform_log_prob(vocab, seq_len, prompt_len):
    """Under a uniform distribution, sum_log_probs = num_completion * log(1/vocab)."""
    logits = _uniform_logits(seq_len, vocab)
    token_ids = torch.zeros(1, seq_len, dtype=torch.long)
    result = _run(logits, token_ids, prompt_len)

    num_completion = seq_len - prompt_len
    expected = num_completion * math.log(1.0 / vocab)
    assert result.item() == pytest.approx(expected, rel=1e-5)


@uniform_cases
def test_result_is_scalar(vocab, seq_len, prompt_len):
    logits = _uniform_logits(seq_len, vocab)
    token_ids = torch.zeros(1, seq_len, dtype=torch.long)
    result = _run(logits, token_ids, prompt_len)
    assert result.shape == torch.Size([])


@uniform_cases
def test_log_prob_is_non_positive(vocab, seq_len, prompt_len):
    logits = _uniform_logits(seq_len, vocab)
    token_ids = torch.zeros(1, seq_len, dtype=torch.long)
    result = _run(logits, token_ids, prompt_len)
    assert result.item() <= 0.0


@temperature_cases
def test_temperature_effect(temperature, token_id, expected_higher):
    """Peaked logits at token 0: low temp -> higher log-prob for that token."""
    vocab, seq_len, prompt_len = 5, 4, 2
    logits = torch.zeros(1, seq_len, vocab)
    logits[0, :, 0] = 10.0  # strongly peaked at token 0
    token_ids = torch.full((1, seq_len), token_id, dtype=torch.long)

    result = _run(logits, token_ids, prompt_len, temperature=temperature)
    # baseline: temperature=1.0
    baseline = _run(logits, token_ids, prompt_len, temperature=1.0)

    if expected_higher:
        assert result.item() > baseline.item()
    else:
        assert result.item() < baseline.item()


def test_temperature_one_is_identity():
    """temperature=1.0 must not change the logits."""
    vocab, seq_len, prompt_len = 6, 5, 2
    logits = torch.randn(1, seq_len, vocab)
    token_ids = torch.randint(0, vocab, (1, seq_len))
    r1 = _run(logits, token_ids, prompt_len, temperature=1.0)
    r2 = _run(logits, token_ids, prompt_len, temperature=1.0)
    assert r1.item() == pytest.approx(r2.item(), rel=1e-6)


@determinism_cases
def test_deterministic(vocab, seq_len, prompt_len):
    torch.manual_seed(0)
    logits = torch.randn(1, seq_len, vocab)
    token_ids = torch.randint(0, vocab, (1, seq_len))
    r1 = _run(logits, token_ids, prompt_len)
    r2 = _run(logits, token_ids, prompt_len)
    assert r1.item() == pytest.approx(r2.item(), rel=1e-7)


@model_call_cases
def test_model_called_exactly_once(vocab, seq_len, prompt_len):
    logits = _uniform_logits(seq_len, vocab)
    token_ids = torch.zeros(1, seq_len, dtype=torch.long)
    model = _make_model(logits)
    compute_log_probs(model, token_ids, prompt_len)
    model.assert_called_once_with(token_ids)
