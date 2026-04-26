import pytest

from tiny_reasoning_model.expression.expression import is_tuple_format


valid_cases = pytest.mark.parametrize("text", [
    pytest.param("(1,2)",         id="simple tuple"),
    pytest.param("(1, 2, 3)",     id="tuple with spaces"),
    pytest.param("(x)",           id="single element"),
    pytest.param("((1,2),(3,4))", id="nested parens inside"),
    pytest.param("(5,\\infty)",   id="latex interval-like"),
    pytest.param("()",            id="empty parens"),
])

invalid_cases = pytest.mark.parametrize("text", [
    pytest.param("1,2",        id="no parens"),
    pytest.param("1,2)",       id="missing opening paren"),
    pytest.param("(1,2",       id="missing closing paren"),
    pytest.param("(1,2)(3,4)", id="outer closes early"),
    pytest.param("(a)(b)",     id="outer closes at middle"),
    pytest.param("(",          id="only opening paren"),
    pytest.param(")",          id="only closing paren"),
    pytest.param("",           id="empty string"),
])


@valid_cases
def test_is_tuple_format_valid(text):
    assert is_tuple_format(text) is True


@invalid_cases
def test_is_tuple_format_invalid(text):
    assert is_tuple_format(text) is False
