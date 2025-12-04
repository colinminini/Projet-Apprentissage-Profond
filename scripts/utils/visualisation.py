"""Generate quick visualizations for a random track to use in the README."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_CSV = Path("data/labeled_tracks.csv")
DEFAULT_MELS_ROOT = Path("data/mels")
DEFAULT_OUTPUT_DIR = Path("sources/visualisation")
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_HOP_LENGTH = 512


def pick_random_track(csv_path: Path, rng: np.random.Generator) -> pd.Series:
    """Return a random row from the labeled tracks CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in CSV: {csv_path}")

    idx = int(rng.integers(0, len(df)))
    return df.iloc[idx]


def load_mel(path: Path) -> np.ndarray:
    """Load a cached mel spectrogram .npy file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing mel file at {path}. Generate mels with scripts/utils/mels.py first."
        )
    return np.load(path)


def plot_mel(
    mel: np.ndarray,
    sample_rate: int,
    hop_length: int,
    title: str,
    output_path: Path,
) -> None:
    """Plot a single mel spectrogram."""
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap="magma",
    )
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_mel_grid(
    mels: Iterable[np.ndarray],
    sample_rate: int,
    hop_length: int,
    title: str,
    output_path: Path,
) -> None:
    """Plot a grid of 10 three-second mels."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey=True)
    axes = axes.ravel()

    for i, mel in enumerate(mels):
        ax = axes[i]
        img = librosa.display.specshow(
            mel,
            sr=sample_rate,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            ax=ax,
            cmap="magma",
        )
        ax.set_title(f"Slice {i + 1}")

    fig.suptitle(title)
    fig.colorbar(img, ax=axes.tolist(), format="%+2.0f dB", shrink=0.6, pad=0.02) # type: ignore
    fig.tight_layout(rect=[0, 0, 1, 0.96]) # type: ignore
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_waveform(
    audio: np.ndarray,
    sample_rate: int,
    title: str,
    output_path: Path,
) -> None:
    """Plot the raw waveform of the selected audio file."""
    fig, ax = plt.subplots(figsize=(12, 3))
    times = np.arange(len(audio)) / sample_rate
    ax.plot(times, audio, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot waveform and mel spectrograms for a random labeled track."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to labeled_tracks.csv.",
    )
    parser.add_argument(
        "--mels-root",
        type=Path,
        default=DEFAULT_MELS_ROOT,
        help="Directory containing mel .npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures will be saved.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Sample rate to load waveform (should match mel generation).",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=DEFAULT_HOP_LENGTH,
        help="Hop length used when mel files were generated.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Optional seed for reproducible track selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    row = pick_random_track(args.csv, rng)
    track_id = int(row["track_id"])
    genre = row.get("genre", "unknown")
    audio_path = Path(row["file_path"])
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Selected track {track_id} (genre: {genre})")

    full_mel_path = args.mels_root / f"{track_id}.npy"
    mel_full = load_mel(full_mel_path)
    plot_mel(
        mel=mel_full,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        title=f"Log-mel spectrogram (30s) - Track {track_id}",
        output_path=args.output_dir / f"{track_id}_mel_30s.png",
    )

    subset_paths = [args.mels_root / f"{track_id}_{i}.npy" for i in range(1, 11)]
    subset_mels = [load_mel(p) for p in subset_paths]
    plot_mel_grid(
        mels=subset_mels,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        title=f"Log-mel spectrograms (3s slices) - Track {track_id}",
        output_path=args.output_dir / f"{track_id}_mel_3s_grid.png",
    )

    audio, _ = librosa.load(audio_path, sr=args.sample_rate)
    plot_waveform(
        audio=audio,
        sample_rate=args.sample_rate,
        title=f"Waveform - Track {track_id}",
        output_path=args.output_dir / f"{track_id}_waveform.png",
    )

    print("Saved figures to:")
    print(f"- {args.output_dir / f'{track_id}_mel_30s.png'}")
    print(f"- {args.output_dir / f'{track_id}_mel_3s_grid.png'}")
    print(f"- {args.output_dir / f'{track_id}_waveform.png'}")


if __name__ == "__main__":
    main()
