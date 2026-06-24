"""Load config.yaml and .env, resolve data paths, detect real vs. synthetic mode."""

from pathlib import Path
import yaml
from dotenv import load_dotenv
import os

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Path | None = None) -> dict:
    load_dotenv(_PROJECT_ROOT / ".env")

    path = config_path or _PROJECT_ROOT / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Environment variables override config.yaml
    if os.getenv("MIMIC_IV_DIR"):
        cfg["data"]["mimic_iv_dir"] = os.getenv("MIMIC_IV_DIR")
    if os.getenv("MIMIC_IV_NOTE_DIR"):
        cfg["data"]["mimic_iv_note_dir"] = os.getenv("MIMIC_IV_NOTE_DIR")

    return cfg


def has_real_data(cfg: dict) -> bool:
    mimic_dir = cfg["data"].get("mimic_iv_dir", "")
    return bool(mimic_dir) and Path(mimic_dir).exists()


def get_data_dir() -> Path:
    return _PROJECT_ROOT / "data"


def get_model_dir() -> Path:
    return _PROJECT_ROOT / "models"
