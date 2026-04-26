import logging
from pathlib import Path

import typer

from tiny_reasoning_model.schema.config import ConfigGRPO
from tiny_reasoning_model.trainer.grpo import GRPO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main(config: Path = typer.Argument(..., help="Path to YAML config file")):
    cfg = ConfigGRPO.from_yaml(config)
    trainer = GRPO(cfg)
    trainer.train()


if __name__ == "__main__":
    typer.run(main)
