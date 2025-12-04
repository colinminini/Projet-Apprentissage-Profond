"""Compute and save log-mel spectrograms for labeled tracks."""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

DEFAULT_INPUT_CSV = Path("data/metadata_ensemble.csv")
DEFAULT_FULL_INPUT_CSV = Path("data/metadata_30_secs.csv")
DEFAULT_OUTPUT_ROOT = Path("data/mels")


def load_audio_clip(
    file_path: Path,
    clip_duration: float,
    sample_rate: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Load audio, then pad or crop to a fixed clip duration."""
    audio, _ = librosa.load(file_path, sr=sample_rate)
    clip_length = int(clip_duration * sample_rate)

    if len(audio) < clip_length:
        audio = np.pad(audio, (0, clip_length - len(audio)), mode="constant")
    elif len(audio) > clip_length:
        start = 0
        if rng is not None:
            start = int(rng.integers(0, len(audio) - clip_length + 1))
        audio = audio[start : start + clip_length]

    return audio


def compute_log_mel(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> np.ndarray:
    """Compute a normalized log-mel spectrogram from an audio vector."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel).astype(np.float32)
    return (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)


def build_output_path(row: pd.Series, output_root: Path) -> Path:
    """Derive the output .npy path for a row."""
    subset = int(row["num_audio_subset"])
    track_id = str(row["track_id"])
    return output_root / f"{track_id}_{subset}.npy"


def expand_seed(base_seed: int | None, track_id: int) -> int | None:
    """Create a deterministic seed per track if a base seed is provided."""
    if base_seed is None:
        return None
    return base_seed + track_id


def process_rows(
    df: pd.DataFrame,
    output_root: Path,
    clip_duration: float,
    subset_duration: float,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    overwrite: bool,
    log_every: int,
    base_seed: int | None,
) -> None:
    processed = 0
    skipped = 0

    subset_len = int(subset_duration * sample_rate)

    for track_id, group in df.groupby("track_id", sort=False):
        audio_path = Path(group.iloc[0]["file_path"])
        if not audio_path.exists():
            print(f"[warn] Missing audio file: {audio_path}")
            skipped += len(group)
            continue

        rng = np.random.default_rng(expand_seed(base_seed, int(track_id))) # type: ignore
        clip = load_audio_clip(
            audio_path,
            clip_duration=clip_duration,
            sample_rate=sample_rate,
            rng=rng,
        )

        for _, row in group.iterrows():
            subset = int(row["num_audio_subset"])
            output_path = build_output_path(row, output_root)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.exists() and not overwrite:
                skipped += 1
                continue

            start = (subset - 1) * subset_len
            end = start + subset_len
            if end > len(clip):
                segment = np.pad(clip[start:], (0, end - len(clip)), mode="constant")
            else:
                segment = clip[start:end]

            mel = compute_log_mel(
                segment,
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )
            np.save(output_path, mel)
            processed += 1

            if processed % log_every == 0:
                print(f"Saved {processed} mels (skipped {skipped})")

    print(f"Finished. Saved: {processed}, skipped: {skipped}, total rows: {len(df)}")


def process_full_rows(
    df: pd.DataFrame,
    output_root: Path,
    clip_duration: float,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    overwrite: bool,
    log_every: int,
    base_seed: int | None,
) -> None:
    processed = 0
    skipped = 0

    for _, row in df.iterrows():
        track_id = int(row["track_id"])
        audio_path = Path(row["file_path"])

        if not audio_path.exists():
            print(f"[warn] Missing audio file: {audio_path}")
            skipped += 1
            continue

        rng = np.random.default_rng(expand_seed(base_seed, track_id))
        clip = load_audio_clip(
            audio_path,
            clip_duration=clip_duration,
            sample_rate=sample_rate,
            rng=rng,
        )

        output_path = output_root / f"{track_id}.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not overwrite:
            skipped += 1
            continue

        mel = compute_log_mel(
            clip,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        np.save(output_path, mel)
        processed += 1

        if processed % log_every == 0:
            print(f"Saved {processed} full-clip mels (skipped {skipped})")

    print(
        f"Finished full-clip pass. Saved: {processed}, skipped: {skipped}, total rows: {len(df)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and cache normalized log-mel spectrograms for labeled tracks."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="CSV with track_id, file_path, num_audio_subset columns.",
    )
    parser.add_argument(
        "--full-csv",
        type=Path,
        default=DEFAULT_FULL_INPUT_CSV,
        help="CSV with track_id, file_path columns for full 30s clips.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where mel .npy files will be stored.",
    )
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=30.0,
        help="Duration (seconds) of the padded/cropped audio clip.",
    )
    parser.add_argument(
        "--subset-duration",
        type=float,
        default=3.0,
        help="Duration (seconds) of each subset slice (default 3s).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Target sample rate for loading audio.",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=2048,
        help="FFT window size for mel computation.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Hop length for mel computation.",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=128,
        help="Number of mel bands.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite existing mel files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for deterministic 30s crops per track when audio is longer than clip-duration.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Log progress every N saved files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows (useful for quick tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    required_cols = {"track_id", "file_path", "num_audio_subset"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    max_subset = int(df["num_audio_subset"].max())
    required_duration = args.subset_duration * max_subset
    if args.clip_duration + 1e-6 < required_duration:
        raise ValueError(
            f"clip-duration must cover all subsets: need at least {required_duration} seconds "
            f"for max subset {max_subset} with subset-duration {args.subset_duration}s"
        )

    if args.limit is not None:
        df = df.head(args.limit)

    process_rows(
        df=df,
        output_root=args.output_root,
        clip_duration=args.clip_duration,
        subset_duration=args.subset_duration,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        overwrite=args.overwrite,
        log_every=args.log_every,
        base_seed=args.seed,
    )

    if args.full_csv is not None and args.full_csv.exists():
        df_full = pd.read_csv(args.full_csv)
        required_full_cols = {"track_id", "file_path"}
        missing_full = required_full_cols - set(df_full.columns)
        if missing_full:
            raise ValueError(f"Missing required columns in full-clip CSV: {missing_full}")

        if args.limit is not None:
            df_full = df_full.head(args.limit)

        process_full_rows(
            df=df_full,
            output_root=args.output_root,
            clip_duration=args.clip_duration,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            overwrite=args.overwrite,
            log_every=args.log_every,
            base_seed=args.seed,
        )
    else:
        print(f"[warn] Full-clip CSV not found at {args.full_csv}, skipping 30s mels.")


if __name__ == "__main__":
    main()
