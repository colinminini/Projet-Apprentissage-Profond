import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from scripts.dataset import Audio_Classification_Dataset


def make_loaders(metadata_path: str, batch_size: int, mel_dir: str = "data/mels"):
    """
    Build train/test dataloaders from a metadata CSV.

    The split keeps ordering by default to preserve 30s clip boundaries used for the ensemble.
    """
    df_metadata = pd.read_csv(metadata_path)
    train_df, test_df = train_test_split(
        df_metadata,
        test_size=0.2,
        shuffle=False,
    )

    train_dataset = Audio_Classification_Dataset(train_df, MEL_DIR=mel_dir)
    test_dataset = Audio_Classification_Dataset(test_df, MEL_DIR=mel_dir)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, test_loader, len(train_df), len(test_df)
