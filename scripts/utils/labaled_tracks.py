"""Step 1 preprocessing: build the labeled track dataframe from the FMA metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

# Default locations mirror the notebook paths
DEFAULT_META_PATH = Path("data/fma_metadata")
DEFAULT_AUDIO_PATH = Path("data/fma_small")
# These 3 first files are known to be corrupted in the FMA small split
DEFAULT_CORRUPTED_TRACK_IDS = (99134, 108925, 133297)
# + Remove 7 more tracks randomly to ensure dataset size is 7990
DEFAULT_CORRUPTED_TRACK_IDS += (69567, 42375, 11795, 120323, 58061, 6358, 75786)


def load_tracks(meta_dir: Path = DEFAULT_META_PATH) -> pd.DataFrame:
    """Load the multi-index FMA tracks metadata CSV."""
    tracks_path = Path(meta_dir) / "tracks.csv"
    if not tracks_path.exists():
        raise FileNotFoundError(f"tracks.csv not found at {tracks_path}")
    return pd.read_csv(tracks_path, index_col=0, header=[0, 1])


def build_dataframe(
    meta_dir: Path = DEFAULT_META_PATH,
    audio_dir: Path = DEFAULT_AUDIO_PATH,
    drop_missing_audio: bool = True,
    corrupted_track_ids: Iterable[int] = DEFAULT_CORRUPTED_TRACK_IDS,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """
    Construct the track dataframe with genre labels and audio file paths.

    Returns (df, genre_to_class, class_to_genre) where:
    - df columns: genre_top, file_path (str), class (int)
    - genre_to_class: mapping from genre label to integer class
    - class_to_genre: inverse mapping
    """
    tracks = load_tracks(meta_dir)

    # Keep only the genre labels columns from the multi-index frame
    df = tracks["track"][["genre_top", "genres", "genres_all"]].copy()
    df = df.dropna(subset=["genre_top"]) # type: ignore
    df = df.drop(columns=["genres", "genres_all"])

    audio_dir = Path(audio_dir)

    def to_audio_path(track_id: int) -> Path:
        tid_str = f"{int(track_id):06d}"
        return audio_dir / tid_str[:3] / f"{tid_str}.mp3"

    df["file_path"] = df.index.map(to_audio_path)

    if drop_missing_audio:
        df = df[df["file_path"].map(Path.exists)]

    corrupted_set = {int(tid) for tid in corrupted_track_ids}
    if corrupted_set:
        df = df[~df.index.isin(corrupted_set)]

    genre_to_class = {genre: idx for idx, genre in enumerate(df["genre_top"].unique())}
    class_to_genre = {idx: genre for genre, idx in genre_to_class.items()}

    df["class"] = df["genre_top"].map(genre_to_class)
    df.rename(columns={"genre_top": "genre"}, inplace=True)
    df["file_path"] = df["file_path"].astype(str)

    return df, genre_to_class, class_to_genre


def main() -> None:
    df, genre_to_class, _ = build_dataframe()
    # Save df as a csv file in the data/ folder
    output_path = "data/labeled_tracks.csv"
    df.to_csv(output_path, index_label="track_id")
    print(f"Labeled track dataframe saved to {output_path}")
    print(f"Tracks kept: {len(df)}")

if __name__ == "__main__":
    main()