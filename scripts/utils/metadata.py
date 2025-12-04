"""Expand labeled_tracks.csv by duplicating rows with mel-spectrogram paths."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("data/labeled_tracks.csv")
DEFAULT_OUTPUT = Path("data/metadata_ensemble.csv")
DEFAULT_OUTPUT_FULL = Path("data/metadata_30_secs.csv")


def expand_labeled_tracks(
    input_path: Path,
    output_path: Path,
    full_output_path: Path,
    seed: int | None = None,
) -> None:
    """
    Shuffle tracks and generate two metadata files:
    - metadata.csv for 3-second slices (10 per track)
    - metadata_30_secs.csv for full 30-second clips (1 per track)
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if "track_id" not in df.columns:
        raise ValueError("Input CSV must include a 'track_id' column.")

    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    subset_values = list(range(1, 11))
    expanded = shuffled.loc[shuffled.index.repeat(len(subset_values))].reset_index(drop=True)
    expanded["num_audio_subset"] = subset_values * len(shuffled)
    expanded["mel_spec_file_path"] = (
        "data/mels/"
        + expanded["track_id"].astype(str)
        + "_"
        + expanded["num_audio_subset"].astype(str)
        + ".npy"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_csv(output_path, index=False)
    print(f"Wrote expanded CSV to {output_path}")
    print(f"Source rows: {len(shuffled)}, expanded rows: {len(expanded)}")

    full = shuffled.copy()
    full["mel_spec_file_path"] = "data/mels/" + full["track_id"].astype(str) + ".npy"
    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(full_output_path, index=False)
    print(f"Wrote full-clip CSV to {full_output_path}")
    print(f"Full-clip rows: {len(full)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shuffle labeled_tracks.csv and expand rows with mel-spectrogram paths."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to labeled_tracks.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path for the expanded dataframe",
    )
    parser.add_argument(
        "--full-output",
        type=Path,
        default=DEFAULT_OUTPUT_FULL,
        help="Output CSV path for the full 30-second clip metadata",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Optional random seed for reproducible shuffling",
    )

    args = parser.parse_args()
    expand_labeled_tracks(args.input, args.output, args.full_output, args.seed)


if __name__ == "__main__":
    main()
