"""Train Stage 1 classical ML models for readmission prediction."""

from pathlib import Path

import pandas as pd

from src.config import load_config, get_data_dir, get_model_dir


def train(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()

    # TODO: implement model training
    raise NotImplementedError("Model training not yet implemented.")


if __name__ == "__main__":
    train()
