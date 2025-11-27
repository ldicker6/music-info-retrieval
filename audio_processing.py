import librosa
import numpy as np

def extract_features(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

    return {
        'chroma': chroma,
        'centroid': spectral_centroid,
        'bandwidth': spectral_bandwidth,
        'rolloff': spectral_rolloff,
        'contrast': spectral_contrast,
        'tempo': tempo
    }
