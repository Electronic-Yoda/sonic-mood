import torchaudio, torch
import numpy as np

class RawAudioToMelspecTransform():
    def __init__(self, n_mels=64, sample_rate=16000, n_fft=400, max_audio_length_sec=3.65,
                test_mode=False) -> None:
        
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
        )
        self.normalize = torchaudio.transforms.AmplitudeToDB()

        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = self.transform.hop_length

        self.max_timeframes  = self._get_max_timeframes(max_audio_length_sec)
        self.max_audio_length_sec = max_audio_length_sec

        self.test_mode = test_mode

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


    def __call__(self, waveform, waveform_sample_rate):
        if self.test_mode:
            print('resample if necessary')
        waveform = self._resample_if_necessary(waveform, waveform_sample_rate)
        if self.test_mode:
            print('mix down if necessary')
        waveform = self._mix_down_if_necessary(waveform)
        if self.test_mode:
            print('truncate or pad waveform')
        waveform = self._truncate_or_pad_waveform(waveform)

        # Apply the transform
        if self.test_mode:
            print('apply transform')
        input = self.transform(waveform)

        if self.test_mode:
            print('normalize')
        input = self.normalize(input)

        if input.shape[2] > self.max_timeframes: # handles the case where the last window starts at the last sample
            input = input[:, :, :self.max_timeframes]

        return input