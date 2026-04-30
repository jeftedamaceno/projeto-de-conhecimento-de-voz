import os
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = "dataset"
OUTPUT_CSV = "dataset.csv"

data = []

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        file_path = os.path.join(label_path, file)

        try:
            audio, sr = librosa.load(file_path, sr=16000)

      
            duration = librosa.get_duration(y=audio, sr=sr)
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            rms = np.mean(librosa.feature.rms(y=audio))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

         
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfcc.T, axis=0)

            row = {
                "caminho": file_path,
                "label": label,
                "duracao": duration,
                "sample_rate": sr,
                "zcr": zcr,
                "rms": rms,
                "spectral_centroid": spectral_centroid
            }

       
            for i in range(13):
                row[f"mfcc_{i+1}"] = mfcc_means[i]

            data.append(row)

        except Exception as e:
            print(f"Erro em {file_path}: {e}")

df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)

print("CSV criado com sucesso!")