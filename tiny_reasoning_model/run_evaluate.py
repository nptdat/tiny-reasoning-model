"""
Usage:
uv run python -m tiny_reasoning_model.run_evaluate config/<config_name>.yaml
"""

import logging
from pathlib import Path

import time
import typer

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from tiny_reasoning_model.schema import Config
from tiny_reasoning_model.evaluator import Grader, Evaluator
from tiny_reasoning_model.utils import get_device

from tiny_reasoning_model.expression import extract_answer
from tiny_reasoning_model.expression.expression import _extract_last_boxed
from tiny_reasoning_model.generation import generate_solution, LLMReasoner


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main(config_file: Path = typer.Argument(..., help="Path to YAML evaluation config file")):
    start_time = time.perf_counter()
    logger.info(
        f"----- Run reasoning for the file {config_file} -----"
    )
    config = Config.from_yaml(config_file)
    logger.info(config)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForCausalLM.from_pretrained(config.model_path)
    model = model.to(device)

    reasoner = LLMReasoner(model, tokenizer, config)
    evaluator = Evaluator()
    result = evaluator.evaluate(
        reasoner, config.dataset_path, config.max_samples, seed=config.seed
    )
    evaluator.save_results(result, config.result_path)

    end_time = time.perf_counter()
    logger.info(f"Finished evaluation for {config_file} in {end_time - start_time} seconds")


if __name__ == "__main__":
    typer.run(main)
