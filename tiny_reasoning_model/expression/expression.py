from __future__ import annotations

import logging
import re
from typing import Union

from sympy import sympify, simplify, Matrix
from sympy.core.expr import Expr
from sympy.matrices.dense import MutableDenseMatrix
from sympy.parsing.latex import parse_latex


RE_REMOVE = re.compile(r"\\left\s*|\\right\s*")
RE_PYTHON_FLOAT = re.compile(r"^\.\d+$")
RE_UNITS_FIRST = re.compile(r"^(\D+)(\d+[\.\d+]*)$")
RE_UNITS_LAST = re.compile(r"^(\d+[\.\d+]*)(\D+)$")


logger = logging.getLogger(__name__)


def extract_answer(text: str) -> str | None:
    return _extract_last_boxed(text)


def _extract_last_boxed(text: str, tag: str = "boxed") -> str | None:
    """Manually extract the last boxed answer because Python basic regex cannot handle nested braces"""
    n = len(text)

    cursor = text.rfind(rf"\{tag}")
    if cursor == -1:
        return None
    while cursor < n and text[cursor] != "{":
        cursor += 1

    cursor += 1
    count = 1
    start = cursor
    while cursor < n and count > 0:
        if text[cursor] == "{":
            count += 1
        elif text[cursor] == "}":
            count -= 1
        cursor += 1
    if count > 0:
        return None

    return text[start : cursor - 1]


def normalize_latex(latex: str) -> str:
    latex = latex.strip()

    # remove
    latex = re.sub(RE_REMOVE, "", latex)

    # transform

    # add 0 to .###
    if RE_PYTHON_FLOAT.match(latex):
        latex = "0" + latex

    latex = latex.strip()
    return latex


def is_tuple_format(text: str) -> bool:
    cursor = 0
    if not text.startswith("(") or not text.endswith(")"):
        return False

    n = len(text)
    cursor = 1
    count = 1
    while cursor < n and count > 0:
        if text[cursor] == "(":
            count += 1
        elif text[cursor] == ")":
            count -= 1
        cursor += 1
    if count == 0 and cursor < n:
        return False
    else:
        return True


class ExpressionBuilder:
    def __init__(self, expression: str) -> None:
        self.expression = expression

    @classmethod
    def from_latex(cls, latex: str) -> "Expression":
        latex = normalize_latex(latex)

        expr = TextExpression.from_latex(latex)

        if expr is None:
            expr = SympyExpression.from_latex(latex)
        if expr is None:
            expr = MatrixExpression.from_latex(latex)
        if expr is None:
            expr = UnitExpression.from_latex(latex)
        if expr is None:
            expr = RangeExpression.from_latex(latex)

        return expr


class TextExpression:
    """Text expression"""

    def __init__(self, text: str) -> None:
        self.text = text

    @classmethod
    # def from_latex(cls, latex: str) -> Union["TextExpression", None]:
    def from_latex(cls, latex: str) -> TextExpression | None:
    
        try:
            # extract value inside \text
            latex = latex.strip()
            if latex.startswith(r"\text"):
                text = _extract_last_boxed(latex, tag="text")
                return cls(text)
        except:
            pass

        return None

    def __eq__(self, other: TextExpression) -> bool:
        if not isinstance(other, TextExpression):
            return False
        return self.text == other.text


class SympyExpression:
    """Expressions which are evaluated with sympy.parse_latex"""

    def __init__(self, expr: Expr) -> None:
        self.expr = expr

    @classmethod
    def from_latex(cls, latex: str) -> "SympyExpression" | None:
        try:
            expr = parse_latex(latex)
            return cls(expr)
        except:
            return None

    def __eq__(self, other: SympyExpression) -> bool:
        if not isinstance(other, SympyExpression):
            return False

        exprs = []
        for expr in [self.expr, other.expr]:
            # In case "-\\dfrac{33}{2}" vs "k = \\frac{-33}{2}" (Equality),
            # compare "-\\dfrac{33}{2}" and "\\frac{-33}{2}"
            # (use the side of the Equality which is not Symbol)
            if expr.is_Equality:
                for expr_ in [expr.lhs, expr.rhs]:
                    if not expr_.is_Symbol:
                        exprs.append(expr_)
            else:
                exprs.append(expr)

        if len(exprs) != 2:
            return False

        try:
            compare_result = simplify(exprs[0] - exprs[1])
        except Exception as e:
            logger.warning(f"Cannot compare {self.expr} and {other.expr}: {e}")
            return False

        return compare_result == 0


class MatrixExpression:
    """Expression of matrices or vectors, e.g. `(1,2)`, `\begin{pmatrix} -2 \\ -14 \\ -7 \end{pmatrix}`"""

    def __init__(self, matrix: MutableDenseMatrix) -> None:
        self.matrix = matrix

    @classmethod
    def from_latex(cls, latex: str) -> "MatrixExpression" | None:
        if "pmatrix" in latex:
            latex = (
                latex.replace(r"\begin{pmatrix}", "")
                .replace(r"\end{pmatrix}", "")
                .strip()
            )
            # split rows on LaTeX row separator \\
            rows = []
            for x in latex.split(r"\\"):
                row = [sympify(y.strip()) for y in x.split("&")]
                rows.append(row)
            matrix = Matrix(rows)
            return cls(matrix)
        elif is_tuple_format(latex):
            latex = latex[1:-1]  # remove ()
            rows = [parse_latex(x.strip()) for x in latex.split(",")]
            matrix = Matrix(rows)
            return cls(matrix)
        return None

    def __eq__(self, other: MatrixExpression) -> bool:
        if not isinstance(other, MatrixExpression):
            return False
        return self.matrix == other.matrix

    def __str__(self) -> str:
        return str(self.matrix)

    def __repr__(self) -> str:
        return repr(self.matrix)


class UnitExpression:
    """Expression with unit, e.g. $18.90, 36°, 36 degree"""

    def __init__(self, expr: Expr, unit: str) -> None:
        self.expr = expr
        self.unit = unit

    @classmethod
    def from_latex(cls, latex: str) -> "UnitExpression" | None:
        m = re.search(RE_UNITS_FIRST, latex)
        if m:
            expr = parse_latex(m.group(2))
            unit = m.group(1)
            return cls(expr, unit)
        else:
            m = re.search(RE_UNITS_LAST, latex)
            if m:
                expr = parse_latex(m.group(1))
                unit = m.group(2)
                unit = unit.strip()
                return cls(expr, unit)

        return None

    def __eq__(self, other: UnitExpression) -> bool:
        if not isinstance(other, UnitExpression):
            return False
        return simplify(self.expr - other.expr) == 0 and self.unit == other.unit


class RangeExpression:
    """Range of values, e.g. (-\infty, 0], (3,4], (2,12)"""

    def __init__(
        self, left_include: bool, left_expr: Expr, right_include: bool, right_expr: Expr
    ) -> None:
        self.left_include = left_include
        self.left_expr = left_expr
        self.right_include = right_include
        self.right_expr = right_expr

    @classmethod
    def from_latex(cls, latex: str) -> "RangeExpression" | None:
        try:
            if (latex.startswith("(") or latex.startswith("[")) and (
                latex.endswith(")") or latex.endswith("]")
            ):

                parts = latex[1:-1].split(",")
                if len(parts) != 2:
                    return None

                left_expr = parse_latex(parts[0].strip())
                right_expr = parse_latex(parts[1].strip())
                left_include = latex.startswith("[")
                right_include = latex.endswith("]")

                return cls(left_include, left_expr, right_include, right_expr)
        except:
            return None

        return None

    def __eq__(self, other: RangeExpression) -> bool:
        if not isinstance(other, RangeExpression):
            return False
        return (
            simplify(self.left_expr - other.left_expr) == 0
            and simplify(self.right_expr - other.right_expr) == 0
            and self.left_include == other.left_include
            and self.right_include == other.right_include
        )

    def __str__(self) -> str:
        return f"{'[' if self.left_include else '('}{self.left_expr}, {self.right_expr}{'}' if self.right_include else ')'}"
