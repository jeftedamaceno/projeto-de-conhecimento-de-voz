import os
import numpy as np
from scipy.io.wavfile import read

from utils import (
    preprocess_audio,
    stft_manual,
    mel_filterbank
)

SAMPLE_RATE = 16000
DURATION = 1.2
TARGET_LENGTH = int(SAMPLE_RATE * DURATION)

N_MELS = 128
MAX_LEN = 94


def audio_para_mel(audio, training=False):

    audio = preprocess_audio(audio, TARGET_LENGTH, training)

    spec = stft_manual(audio)
    spec_power = np.abs(spec) ** 2

    mel_fb = mel_filterbank(SAMPLE_RATE, 512, N_MELS)

    mel_spec = np.dot(mel_fb, spec_power.T)

    mel_db = 10 * np.log10(mel_spec + 1e-10)

    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    if mel_db.shape[1] < MAX_LEN:
        mel_db = np.pad(mel_db, ((0,0),(0,MAX_LEN - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :MAX_LEN]

    return mel_db


def processar_dataset(input_dir, output_dir, training=True):

    for label in os.listdir(input_dir):
        in_path = os.path.join(input_dir, label)
        out_path = os.path.join(output_dir, label)

        os.makedirs(out_path, exist_ok=True)

        for file in os.listdir(in_path):

            if file.endswith(".wav"):

                sr, audio = read(os.path.join(in_path, file))

                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)

                mel = audio_para_mel(audio, training)

                np.save(os.path.join(out_path, file.replace(".wav", ".npy")), mel)



processar_dataset("dataset_final", "spectrograms_original", training=False)
processar_dataset("dataset_ruido", "spectrograms_ruido", training=False)