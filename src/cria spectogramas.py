import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = "dataset"
OUTPUT_PATH = "spectrograms"

os.makedirs(OUTPUT_PATH, exist_ok=True)

def save_spectrogram(file_path, output_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)

        # 🔥 gera Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=128
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # salvar como imagem
        plt.figure(figsize=(3, 3))
        plt.axis('off')
        plt.imshow(mel_spec_db, aspect='auto', origin='lower')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    except Exception as e:
        print(f"Erro: {file_path} -> {e}")


# percorrer dataset
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
            os.path.splitext(file)[0] + ".png"
        )
        # output_file = os.path.join(
        #     output_label_path,
        #     file.replace(".ogg", ".png")
        # )

        save_spectrogram(file_path, output_file)

print("Spectrogramas gerados!")