"""Run inference with a trained Stage 1 model."""

from src.config import load_config


def predict(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()

    # TODO: implement prediction
    raise NotImplementedError("Prediction not yet implemented.")


if __name__ == "__main__":
    predict()
