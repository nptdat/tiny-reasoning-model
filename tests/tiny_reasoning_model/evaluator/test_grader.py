import pytest

from tiny_reasoning_model.evaluator.grader import Grader


# --- Correct answers (score 1.0) ---
correct_cases = pytest.mark.parametrize("solution,correct_answer", [
    pytest.param(
        r"The answer is \boxed{42}",
        r"42",
        id="integer match",
    ),
    pytest.param(
        r"Therefore \boxed{\frac{1}{3}}",
        r"\frac{1}{3}",
        id="fraction match",
    ),
    pytest.param(
        r"We get \boxed{0.5}",
        r"\frac{1}{2}",
        id="decimal equals fraction",
    ),
    pytest.param(
        r"Answer: \boxed{x^2 + 2x + 1}",
        r"(x+1)^2",
        id="symbolic equivalence",
    ),
    pytest.param(
        r"The result is \boxed{-\frac{1}{3}}",
        r"-1/3",
        id="negative fraction",
    ),
    pytest.param(
        r"The result is \boxed{\text{Evelyn}}",
        r"\text{Evelyn}",
        id="text match1",
    ),
])

# --- Wrong answers (score 0.0) ---
wrong_cases = pytest.mark.parametrize("solution,correct_answer", [
    pytest.param(
        r"The answer is \boxed{43}",
        r"42",
        id="wrong integer",
    ),
    pytest.param(
        r"Therefore \boxed{\frac{1}{3}}",
        r"\frac{2}{3}",
        id="wrong fraction",
    ),
    pytest.param(
        r"Answer: \boxed{x^2 + 1}",
        r"(x+1)^2",
        id="wrong symbolic",
    ),
    pytest.param(
        r"The result is \boxed{\text{Evelyn}}",
        r"\text{Evelyn1}",
        id="text unmatch1",
    ),
    pytest.param(
        r"The result is \boxed{\text{Evelyn}}",
        r"Evelyn",
        id="text unmatch2",
    ),
])

# --- Edge cases (score 0.0) ---
edge_cases = pytest.mark.parametrize("solution,correct_answer", [
    pytest.param(
        r"No boxed answer here",
        r"42",
        id="missing boxed in solution",
    ),
    pytest.param(
        r"The answer is \boxed{42}",
        r"unparseable???",
        id="unparseable correct_answer",
    ),
    pytest.param(
        r"The answer is \boxed{???unparseable}",
        r"42",
        id="unparseable boxed answer",
    ),
    pytest.param(
        r"",
        r"42",
        id="empty solution",
    ),
])


@correct_cases
def test_grader_correct(solution, correct_answer):
    assert Grader.grade(solution, correct_answer) == 1.0


@wrong_cases
def test_grader_wrong(solution, correct_answer):
    assert Grader.grade(solution, correct_answer) == 0.0


@edge_cases
def test_grader_edge_cases(solution, correct_answer):
    assert Grader.grade(solution, correct_answer) == 0.0
