from tiny_reasoning_model.expression import (
    ExpressionBuilder, extract_answer
)


class Grader:
    def __init__(self) -> None:
        pass

    @classmethod
    def grade(cls, solution: str, correct_answer: str) -> int:
        """
            Arguments:
                solution (str): the full solution to be graded. 
                    The final answer is expected to be enclosed in \boxed{...}
                correct_answer (str): the short correct answer from ground-truth data
            Return:
                1 if the solution is correct, else 0
        """

        answer = extract_answer(solution)
        if answer is None:
            return 0
        
        # exact match (string comparison)
        if answer == correct_answer:
            return 1

        correct_expr = ExpressionBuilder.from_latex(correct_answer)
        if correct_expr is None:
            return 0
        
        answer_expr = ExpressionBuilder.from_latex(answer)
        if answer_expr is None:
            return 0
        
        # symbolic match
        if correct_expr == answer_expr:
            return 1

        return 0
