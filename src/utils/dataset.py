import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import numpy as np
from dataclasses import dataclass

from .constants import Columns, Emotions

    

class AudioEmotionDataset(Dataset):

    def __init__(self, csv_dataframe, n_mels=64, sample_rate=16000, n_fft=400, max_audio_length_sec=3.65,
                test_mode=False) -> None:
        '''
        args:
            csv_dataframe: a pandas dataframe containing the metadata of the dataset
            n_mels: the number of mel bands to generate
            sample_rate: the target sample rate for the audio files. If the audio files have different sample rates, they will be resampled to this rate
            n_fft: the number of fft bins
            max_audio_length_sec: the maximum audio length in seconds
            test_mode: if True, return the waveform and audio file path, and audio_length as well
        '''

        self.dataframe = csv_dataframe
        self.labels = self.dataframe[Columns.EMOTIONS].unique()

        # Create a MelSpectrogram transform by default
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=400,
        )
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = self.transform.hop_length
        
        self.max_timeframes  = self._get_max_timeframes(max_audio_length_sec)
        self.max_audio_length_sec = max_audio_length_sec

        self.test_mode = test_mode

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx) -> tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_file_path = self.dataframe.loc[idx, Columns.PATH]
        audio_length = self.dataframe.loc[idx, Columns.AUDIO_LENGTH]
        waveform, waveform_sample_rate = torchaudio.load(audio_file_path)

        waveform = self._resample_if_necessary(waveform, waveform_sample_rate)
        waveform = self._mix_down_if_necessary(waveform)
        waveform = self._truncate_or_pad_waveform(waveform)

        # Apply the transform
        input = self.transform(waveform)

        # encode the label as a long tensor
        label = Emotions.to_index(self.dataframe.loc[idx, Columns.EMOTIONS])
        label = torch.tensor(label, dtype=torch.long)

        if self.test_mode:
            return input, label, waveform, audio_length, audio_file_path
        else:
            return input, label


    def _get_max_timeframes(self, max_audio_length_sec):
        # Estimate the maximum number of timeframes based on the audio lengths
        max_timeframes = int(np.ceil(max_audio_length_sec * self.sample_rate / self.hop_length))
        return max_timeframes
    
    def _resample_if_necessary(self, waveform, waveform_sample_rate):
        if waveform_sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(waveform_sample_rate, self.sample_rate)(waveform)
        return waveform
    
    def _mix_down_if_necessary(self, waveform):
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
    
    def _truncate_or_pad_waveform(self, waveform):
        max_samples = self.max_timeframes * self.hop_length
        if waveform.shape[1] < max_samples:
            # Pad waveform with zeros
            waveform = torch.nn.functional.pad(waveform, (0, max_samples - waveform.shape[1]))
        else:
            # Truncate waveform
            waveform = waveform[:, :max_samples]
        return waveform