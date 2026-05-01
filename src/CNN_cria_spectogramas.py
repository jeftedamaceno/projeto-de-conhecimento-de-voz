import os
import numpy as np
import soundfile as sf

from utils import stft_manual, mel_filterbank, normalizar

DATASET_PATH = "dataset_final"
OUTPUT_PATH = "spectrograms"

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 128

os.makedirs(OUTPUT_PATH, exist_ok=True)


def pad_or_trim(mel, max_len=94):
    if mel.shape[1] < max_len:
        pad_width = max_len - mel.shape[1]
        mel = np.pad(mel, ((0,0),(0,pad_width)))
    else:
        mel = mel[:, :max_len]
    return mel


def power_to_db(S):
    return 10 * np.log10(S + 1e-10)


def save_spectrogram(file_path, output_path):
    try:
        audio, sr = sf.read(file_path)

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        if sr != SAMPLE_RATE:
            tempo = np.linspace(0, len(audio)/sr, int(len(audio)*SAMPLE_RATE/sr))
            audio = np.interp(tempo, np.linspace(0, len(audio)/sr, len(audio)), audio)

        audio = normalizar(audio)

        spec = stft_manual(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
        spec_power = np.abs(spec) ** 2

        mel_fb = mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)
        # mel_spec = np.dot(mel_fb, spec_power)
        mel_spec = np.dot(mel_fb, spec_power.T)

        mel_db = power_to_db(mel_spec)

        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

        mel_db = pad_or_trim(mel_db)

        np.save(output_path, mel_db)

    except Exception as e:
        print(f"Erro: {file_path} -> {e}")


for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    output_label_path = os.path.join(OUTPUT_PATH, label)
    os.makedirs(output_label_path, exist_ok=True)

    for file in os.listdir(label_path):
        file_path = os.path.join(label_path, file)

        output_file = os.path.join(
            output_label_path,
            os.path.splitext(file)[0] + ".npy"
        )

        save_spectrogram(file_path, output_file)

print("Spectrogramas gerados!")