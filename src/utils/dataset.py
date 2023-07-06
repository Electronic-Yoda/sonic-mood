import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os

from .constants import Columns

class AudioEmotionDataset(Dataset):
    def __init__(self, base_dir, path_to_csv, n_mels=128, sample_rate=16000):
        self.base_dir = base_dir
        self.dataframe = pd.read_csv(path_to_csv)
        self.labels = self.dataframe[Columns.EMOTIONS].unique()

        # Create a MelSpectrogram transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_file_path = os.path.join(self.base_dir, self.dataframe.loc[idx, Columns.PATH])
        waveform, sample_rate = torchaudio.load(audio_file_path)

        # Apply the MelSpectrogram transform
        mel_spec = self.mel_spectrogram(waveform)

        # encode the label as a long tensor
        label = self.labels.tolist().index(self.dataframe.loc[idx, Columns.EMOTIONS])
        label = torch.tensor(label, dtype=torch.long)

        return mel_spec, label
