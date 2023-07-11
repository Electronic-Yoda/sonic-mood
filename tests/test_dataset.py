import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR) # Add src/ directory to python path so we can import our modules

from unittest import TestCase, main
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from src.utils.dataset import AudioEmotionDataset
from src.utils.constants import Columns, Emotions, Datasets
import src.utils.data_processing as data_processing



class AudiEmotionDatasetTestCase(TestCase):
    def test_dataset(self):
        # purpose: test that __getitem__ returns the correct audio file and label
        csv_df = data_processing.read_csv(
            pd.read_csv(
                os.path.join(BASE_DIR, 'data/metadata/dataset.csv'),
                index_col=False
            ),
            BASE_DIR
        )

        dataset = AudioEmotionDataset(csv_df, test_mode=True)

        for i in range(min(len(dataset), 30)):
            input, label, waveform, audio_length, audio_file_path = dataset[i]
            self.assertEqual(audio_file_path, csv_df.loc[i, Columns.PATH])
            self.assertEqual(Emotions().from_index(label.item()), csv_df.loc[i, Columns.EMOTIONS])
            self.assertEqual(audio_length, csv_df.loc[i, Columns.AUDIO_LENGTH])

            # note: won't test mel_spec and waveform here.
        
    def test_transform_padding(self):
        # purpose: test that the transform is applied correctly

        csv_df = data_processing.read_csv(
            pd.read_csv(
                os.path.join(BASE_DIR, 'data/metadata/dataset.csv'),
                index_col=False
            ),
            BASE_DIR
        )

        dataset = AudioEmotionDataset(csv_df, test_mode=True)
        print('\nmax_audio_length_sec', dataset.max_audio_length_sec)
        print('max_timeframes', dataset.max_timeframes)

        for i in range(min(len(dataset), 30)):
            input, label, waveform, audio_length, audio_file_path = dataset[i]

            print(input.shape[1], input.shape[2])
            self.assertEqual(input.shape[1], dataset.n_mels)
            self.assertEqual(input.shape[2], int(np.ceil(dataset.max_audio_length_sec * dataset.sample_rate / dataset.hop_length)))

    def test_transform_spectrogram_plot(self):
        # purpose: test that the transform is applied correctly

        csv_df = data_processing.read_csv(
            pd.read_csv(
                os.path.join(BASE_DIR, 'data/metadata/dataset.csv'),
                index_col=False
            ),
            BASE_DIR
        )

        dataset = AudioEmotionDataset(csv_df, max_audio_length_sec=2.7, test_mode=True)

        for i in range(min(len(dataset), 20)):
            input, label, waveform, audio_length, audio_file_path = dataset[i]

            # Plot the spectrogram
            plt.figure(figsize=(10, 4))
            plt.imshow(input[0].numpy(), cmap='hot', origin='lower', aspect='auto')
            plt.title('Mel Spectrogram')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            # plt.show()
            plt.pause(1)  # display each plot for 1 seconds



if __name__ == '__main__':
    main()