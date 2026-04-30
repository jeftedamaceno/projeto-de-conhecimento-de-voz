import os
import librosa
import numpy as np

INPUT_DIR = "dataset_ruido"
OUTPUT_DIR = "spectrograms_ruido"

SAMPLE_RATE = 16000
N_MELS = 128
MAX_LEN = 94

os.makedirs(OUTPUT_DIR, exist_ok=True)


def processar_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

  
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)


    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

   
    if mel_db.shape[1] < MAX_LEN:
        pad = MAX_LEN - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad)))
    else:
        mel_db = mel_db[:, :MAX_LEN]

    return mel_db


for label in os.listdir(INPUT_DIR):
    input_label = os.path.join(INPUT_DIR, label)

    if not os.path.isdir(input_label):
        continue

    output_label = os.path.join(OUTPUT_DIR, label)
    os.makedirs(output_label, exist_ok=True)

    for file in os.listdir(input_label):
        if not file.endswith(".wav"):
            continue

        try:
            caminho = os.path.join(input_label, file)
            mel = processar_audio(caminho)

            output_file = os.path.join(
                output_label,
                os.path.splitext(file)[0] + ".npy"
            )

            np.save(output_file, mel)

        except Exception as e:
            print(f"Erro: {file} -> {e}")

print("Spectrogramas com ruído gerados!")