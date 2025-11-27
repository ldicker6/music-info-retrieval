import os

def list_audio_files(directory):
    return [f for f in os.listdir(directory) if f.endswith(".wav") or f.endswith(".mp3")]
