import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = "dataset"
OUTPUT_PATH = "spectrograms"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def pad_or_trim(mel, max_len=94):
    if mel.shape[1] < max_len:
        pad_width = max_len - mel.shape[1]
        mel = np.pad(mel, ((0,0),(0,pad_width)))
    else:
        mel = mel[:, :max_len]
    return mel
def save_spectrogram(file_path, output_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=128
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

      
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        mel_spec_db = pad_or_trim(mel_spec_db)
        np.save(output_path, mel_spec_db)

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