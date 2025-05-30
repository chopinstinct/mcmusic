import librosa
import soundfile as sf
import numpy as np
import os

import librosa
import soundfile as sf
import numpy as np
import random

def preprocess_audio(input_path, output_path, sample_rate=22050, duration=5):
    y, sr = librosa.load(input_path, sr=sample_rate, mono=True)
    total_samples = len(y)
    target_samples = duration * sample_rate

    if total_samples <= target_samples:
        # Pad if the file is too short
        y = librosa.util.fix_length(y, target_samples)
    else:
        # Choose a random start index
        max_start = total_samples - target_samples
        start_sample = random.randint(0, max_start)
        y = y[start_sample:start_sample + target_samples]

    sf.write(output_path, y, sample_rate)


input_path = "KaruthavanlaamGaleejamB2W2.mp3"   # Or .wav, etc.
output_path = "processed_audio_2.wav"

preprocess_audio(input_path, output_path)