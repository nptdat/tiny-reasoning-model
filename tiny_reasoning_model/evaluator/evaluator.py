import json
from time import time
from logging import getLogger
from pathlib import Path

from datasets import load_dataset
import torch

from tqdm import tqdm
from .grader import Grader
from tiny_reasoning_model.generation import LLMReasoner
from tiny_reasoning_model.utils import set_seed


logger = getLogger(__name__)


class Evaluator:
    def __init__(self) -> None:
        pass

    def evaluate(
        self,
        reasoner: LLMReasoner,
        dataset_path: str,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> dict[str, float]:
        # Correct for "HuggingFaceH4/MATH-500"
        # TODO: confirm with other datasets
        start_time = time()

        set_seed(seed)

        test_ds = load_dataset(dataset_path)["test"]

        total = 0
        correct = 0
        details = []
        for i, item in enumerate(tqdm(test_ds)):
            try:
                problem = item["problem"]
                correct_answer = item["answer"]
                with torch.inference_mode():
                    solution = reasoner.solve(problem)
                grade = Grader.grade(solution, correct_answer)
                correct += grade
                total += 1
            except Exception as e:
                logger.error(f"Error at {i}: {e}")
                continue

            details.append(
                dict(
                    no=i,
                    problem=problem,
                    correct_answer=correct_answer,
                    solution=solution,
                    grade=grade,
                )
            )

            logger.info(
                f"Problem {i} with {grade=}, {correct=}, {total=}, Accuracy so far: {correct / total}"
            )
            if max_samples is not None and total >= max_samples:
                break

        elapsed_time = time() - start_time
        return dict(
            dataset_path=dataset_path,
            correct=correct,
            total=total,
            accuracy=correct / total,
            elapsed_time=elapsed_time,
            config=reasoner.config.model_dump(),
            details=details,
        )

    def save_results(self, result: dict, result_path: str | Path | None) -> None:
        if result_path is None:
            return
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
