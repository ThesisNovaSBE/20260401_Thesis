"""One-command demo: generate synthetic data and verify the pipeline runs.

Usage:
    python setup_demo.py

This script works WITHOUT real MIMIC-IV data. It generates a synthetic dataset
matching the MIMIC-IV column schema and prints summary statistics.
"""

from pathlib import Path

from src.config import load_config, get_data_dir
from src.data.synthetic import generate_synthetic_dataset


def main():
    cfg = load_config()
    out = get_data_dir() / "synthetic"

    print("=" * 60)
    print("  Readmission Prediction — Demo Setup (Synthetic Data)")
    print("=" * 60)
    print()

    features = generate_synthetic_dataset(
        n_patients=cfg["data"]["synthetic_n_patients"],
        seed=cfg["data"]["synthetic_seed"],
        output_dir=out,
    )

    print()
    print("Feature matrix preview:")
    print(features.head(10).to_string(index=False))
    print()
    print("Column types:")
    print(features.dtypes.to_string())
    print()
    print("Demo complete. You can now develop against data/synthetic/features.csv")


if __name__ == "__main__":
    main()
