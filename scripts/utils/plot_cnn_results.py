"""Plot training/testing metrics from CSV logs and save figures to disk."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_LOG_DIR = Path("sources/visualisation/CNN_results_csv")
DEFAULT_OUTPUT_DIR = Path("sources/visualisation")


def read_metric_csv(path: Path) -> pd.DataFrame:
    """Load a metrics CSV produced by TensorBoard."""
    if not path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"Step", "Value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in CSV: {path}")
    return df


def plot_curves(
    curves: Iterable[dict[str, pd.Series]],
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot multiple curves on the same axes and save to a PNG."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for curve in curves:
        ax.plot(curve["x"], curve["y"], marker="o", label=curve["label"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CNN experiment logs stored as CSV files."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory containing CNN_exp_ensemble_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNGs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logs = {
        "train_accuracy": args.logs_dir / "CNN_exp_ensemble_train_accuracy.csv",
        "test_accuracy": args.logs_dir / "CNN_exp_ensemble_testing_accuracy.csv",
        "train_loss": args.logs_dir / "CNN_exp_ensemble_train_loss.csv",
        "test_loss": args.logs_dir / "CNN_exp_ensemble_testing_loss.csv",
        "train_batch_loss": args.logs_dir / "CNN_exp_ensemble_train_batch_loss.csv",
    }

    train_acc = read_metric_csv(logs["train_accuracy"])
    test_acc = read_metric_csv(logs["test_accuracy"])
    plot_curves(
        curves=[
            {"x": train_acc["Step"], "y": train_acc["Value"], "label": "Train accuracy"},
            {"x": test_acc["Step"], "y": test_acc["Value"], "label": "Test accuracy"},
        ], # type: ignore
        xlabel="Epoch",
        ylabel="Accuracy",
        title="CNN Ensemble Accuracy",
        output_path=args.output_dir / "CNN_exp_ensemble_accuracy.png",
    )

    train_loss = read_metric_csv(logs["train_loss"])
    test_loss = read_metric_csv(logs["test_loss"])
    plot_curves(
        curves=[
            {"x": train_loss["Step"], "y": train_loss["Value"], "label": "Train loss"},
            {"x": test_loss["Step"], "y": test_loss["Value"], "label": "Test loss"},
        ], # type: ignore
        xlabel="Epoch",
        ylabel="Cross-entropy loss",
        title="CNN Ensemble Loss",
        output_path=args.output_dir / "CNN_exp_ensemble_loss.png",
    )

    train_batch_loss = read_metric_csv(logs["train_batch_loss"])
    plot_curves(
        curves=[
            {
                "x": train_batch_loss["Step"],
                "y": train_batch_loss["Value"],
                "label": "Train batch loss", # type: ignore
            }
        ],
        xlabel="Batch step",
        ylabel="Cross-entropy loss",
        title="CNN Ensemble Batch Loss (train)",
        output_path=args.output_dir / "CNN_exp_ensemble_train_batch_loss.png",
    )

    saved = [
        args.output_dir / "CNN_exp_ensemble_accuracy.png",
        args.output_dir / "CNN_exp_ensemble_loss.png",
        args.output_dir / "CNN_exp_ensemble_train_batch_loss.png",
    ]
    print("Saved figures:")
    for path in saved:
        print(f"- {path}")


if __name__ == "__main__":
    main()
