"""Evaluate Stage 1 models — recall, precision, AUROC, confusion matrix."""

from src.config import load_config


def evaluate(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()

    # TODO: implement evaluation
    raise NotImplementedError("Evaluation not yet implemented.")


if __name__ == "__main__":
    evaluate()
