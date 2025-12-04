import numpy as np
import torch
from torch.utils.data import Dataset

class Audio_Classification_Dataset(Dataset):

    def __init__(self, df, MEL_DIR):
      self.df = df
      self.MEL_DIR = MEL_DIR

    def __len__(self):
      return len(self.df)

    def __getitem__(self, idx):

      row = self.df.iloc[idx]
      mel_path = row["mel_spec_file_path"]
      mel_db_norm = np.load(mel_path)
      x_tensor = torch.from_numpy(mel_db_norm).unsqueeze(0) # Size (1, n_mels, time)
      label = int(row['class'])

      return x_tensor, label  # label is an int between 0 and num_classes (here 7)
                              # Input size is always the same : (1, n_mels=128, time)
                              # with time being constant (we padded the audios)
