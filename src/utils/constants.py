# constants.py
    
class Columns():
    PATH = 'Path'
    EMOTIONS = 'Emotions'
    DATASET = 'Dataset'
    AUDIO_LENGTH = 'Audio_Length'

class Emotions:
    NEUTRAL = 'neutral'
    CALM = 'calm'
    HAPPY = 'happy'
    SAD = 'sad'
    ANGRY = 'angry'
    FEAR = 'fear'
    DISGUST = 'disgust'
    SURPRISE = 'surprise'
    
    _emotions = [NEUTRAL, CALM, HAPPY, SAD, ANGRY, FEAR, DISGUST, SURPRISE]

    @classmethod
    def __len__(cls):
        return len(cls._emotions)

    @classmethod
    def __iter__(cls):
        return iter(cls._emotions)

    @classmethod
    def to_list(cls):
        return cls._emotions.copy()



class Datasets():
    RAVDESS = 'RAVDESS'
    SAVEE = 'SAVEE'
    TESS = 'TESS'
    CREMA = 'CREMA'