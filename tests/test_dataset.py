import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR) # Add src/ directory to python path so we can import our modules

from unittest import TestCase, main
import pandas as pd
from src.utils.dataset import AudioEmotionDataset
from src.utils.constants import Columns, Emotions, Datasets
import src.utils.data_processing as dp



class AudiEmotionDatasetTestCase(TestCase):
    def test_dataset(self):
        # purpose: test that __getitem__ returns the correct audio file and label

        csv_df = dp.add_rel_path(
            pd.read_csv(
                os.path.join(BASE_DIR, 'data/metadata/dataset.csv'),
                index_col=False
            ),
            BASE_DIR
        )

        dataset = AudioEmotionDataset(csv_df)

        for i in range(len(dataset)):
            mel_spec, label, waveform, audio_length, audio_file_path = dataset[i]
            self.assertEqual(audio_file_path, csv_df.loc[i, Columns.PATH])
            self.assertEqual(Emotions().from_index(label.item()), csv_df.loc[i, Columns.EMOTIONS])
            self.assertEqual(audio_length, csv_df.loc[i, Columns.AUDIO_LENGTH])

            # note: won't test mel_spec and waveform here. They will be tested by plotting them in the notebook


if __name__ == '__main__':
    main()