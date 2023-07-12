# utils/data_processing.py

import pandas as pd
import torchaudio
import torch
import os
from .constants import Columns, Emotions, Datasets


# Functions to extract path and labels for each dataset

def get_crema(base_dir, path):
    path = os.path.join(base_dir, path)
    emotions = {'ANG': Emotions.ANGRY, 'DIS': Emotions.DISGUST, 'FEA': Emotions.FEAR,
                'HAP': Emotions.HAPPY, 'NEU': Emotions.NEUTRAL, 'SAD': Emotions.SAD}
    file_paths = []
    file_emotions = []
    audio_lengths = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        file_paths.append(os.path.relpath(file_path, base_dir).replace('\\', '/'))
        emotion_code = file.split('_')[2]
        file_emotions.append(emotions[emotion_code])
        audio_lengths.append(get_audio_length(file_path))
    
    return pd.DataFrame({
        Columns.PATH: file_paths,
        Columns.EMOTIONS: file_emotions,
        Columns.DATASET: Datasets.CREMA,
        Columns.AUDIO_LENGTH: audio_lengths
    })



def get_ravdess(base_dir, path):
    path = os.path.join(base_dir, path)
    emotions = {1: Emotions.NEUTRAL, 2: Emotions.CALM, 3: Emotions.HAPPY,
                4: Emotions.SAD, 5: Emotions.ANGRY, 6: Emotions.FEAR,
                7: Emotions.DISGUST, 8: Emotions.SURPRISE}

    file_paths = []
    file_emotions = []
    audio_lengths = []
    for actor_dir in os.listdir(path):
        actor_dir_path = os.path.join(path, actor_dir)
        for filename in os.listdir(actor_dir_path):
            file_path = os.path.join(actor_dir_path, filename)
            part = filename.split('.')[0]
            part = part.split('-')
            emotion_code = int(part[2])
            file_paths.append(os.path.relpath(file_path, base_dir).replace('\\', '/'))
            file_emotions.append(emotions[emotion_code])
            audio_lengths.append(get_audio_length(file_path))
    
    return pd.DataFrame({
        Columns.PATH: file_paths,
        Columns.EMOTIONS: file_emotions,
        Columns.DATASET: Datasets.RAVDESS,
        Columns.AUDIO_LENGTH: audio_lengths
    })


def get_savee(base_dir, path):
    path = os.path.join(base_dir, path)
    emotions = {'a': Emotions.ANGRY,'d': Emotions.DISGUST, 'f': Emotions.FEAR,
                'h': Emotions.HAPPY, 'n': Emotions.NEUTRAL, 'sa': Emotions.SAD,
                'su': Emotions.SURPRISE}

    file_paths = []
    file_emotions = []
    audio_lengths = []

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        file_paths.append(os.path.relpath(file_path, base_dir).replace('\\', '/'))
        part = file.split('_')[1]
        emotion_code = part[:-6]
        file_emotions.append(emotions[emotion_code])
        audio_lengths.append(get_audio_length(file_path))

    return pd.DataFrame({
        Columns.PATH: file_paths,
        Columns.EMOTIONS: file_emotions,
        Columns.DATASET: Datasets.SAVEE,
        Columns.AUDIO_LENGTH: audio_lengths
    })

def get_tess(base_dir, path):
    path = os.path.join(base_dir, path)
    emotions = {'neutral': Emotions.NEUTRAL, 'angry': Emotions.ANGRY,
                'disgust': Emotions.DISGUST, 'ps': Emotions.SURPRISE,
                'happy': Emotions.HAPPY, 'sad': Emotions.SAD,
                'fear': Emotions.FEAR}

    file_paths = []
    file_emotions = []
    audio_lengths = []

    for dir in os.listdir(path):
        for file in os.listdir(os.path.join(path, dir)):
            part = file.split('.')[0]
            emotion_code = part.split('_')[2]
            file_path = os.path.join(path, dir, file)
            file_paths.append(os.path.relpath(file_path, base_dir).replace('\\', '/'))
            file_emotions.append(emotions[emotion_code])
            audio_lengths.append(get_audio_length(file_path))
    
    return pd.DataFrame({
        Columns.PATH: file_paths,
        Columns.EMOTIONS: file_emotions,
        Columns.DATASET: Datasets.TESS,
        Columns.AUDIO_LENGTH: audio_lengths
    })

# Function to get audio length
def get_audio_length(path):
    waveform, sample_rate = torchaudio.load(path)
    return waveform.shape[1] / sample_rate

def read_csv(df, base_dir):
    '''Reads the csv file and returns a dataframe with the path relative to the base_dir'''

    # create a copy of the dataframe
    df_cp = df.copy()
    df_cp[Columns.PATH] = df_cp[Columns.PATH].apply(lambda x: os.path.join(base_dir, x))
    return df_cp


# function to convert the audio files to spectrograms
def convert_to_spectrogram(dataset, BASE_DIR, output_spectrogram_path, output_metadata_path):
    paths = []
    labels = []
    for idx in range(len(dataset)):
        spectrogram, label = dataset[idx]
        rel_path = os.path.join(output_spectrogram_path, f'{idx}.pt')
        path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(spectrogram, path)
        paths.append(rel_path.replace('\\', '/'))
        labels.append(Emotions.from_index(label.item()))

    mel_df = pd.DataFrame({
        Columns.PATH: paths,
        Columns.EMOTIONS: labels,
    })

    mel_df.to_csv(os.path.join(BASE_DIR, output_metadata_path), index=False)
    return mel_df