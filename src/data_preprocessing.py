import librosa
import numpy as np
import os

def preprocess_audio(audio_path):
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Resample to 16 kHz for compatibility with models
    audio = librosa.resample(audio, sr, 16000)
    
    # Normalize audio
    audio = audio / np.max(np.abs(audio))

    # Save preprocessed audio
    preprocessed_path = os.path.join("data/processed", os.path.basename(audio_path))
    librosa.output.write_wav(preprocessed_path, audio, 16000)

    return preprocessed_path
