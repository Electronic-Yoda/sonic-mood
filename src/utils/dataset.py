import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
from dataclasses import dataclass

from .constants import Columns, Emotions

    

class AudioEmotionDataset(Dataset):
    # change so that the csv is passed in as a dataframe
    def __init__(self, csv_dataframe, n_mels=64, sample_rate=16000,
                transform=None, test_mode=False) -> None:

        self.dataframe = csv_dataframe
        self.labels = self.dataframe[Columns.EMOTIONS].unique()

        if transform == None:
            # Create a MelSpectrogram transform by default
            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_fft=400
            )
        else:
            self.transform = transform # use the transform passed in

        self.test_mode = test_mode

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx) -> tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_file_path = self.dataframe.loc[idx, Columns.PATH]
        audio_length = self.dataframe.loc[idx, Columns.AUDIO_LENGTH]
        waveform, sample_rate = torchaudio.load(audio_file_path)

        # Apply the transform
        input = self.transform(waveform)

        # encode the label as a long tensor
        label = Emotions.to_index(self.dataframe.loc[idx, Columns.EMOTIONS])
        label = torch.tensor(label, dtype=torch.long)

        if self.test_mode:
            return input, label, waveform, audio_length, audio_file_path
        else:
            return input, label
