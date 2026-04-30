
import os
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = "dataset"
OUTPUT_CSV_NOISE = "dataset_ruido.csv"

data = []

MAX_AUG_PER_AUDIO = 2


def extract_features(audio, sr):
    duration = librosa.get_duration(y=audio, sr=sr)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    rms = np.mean(librosa.feature.rms(y=audio))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc.T, axis=0)

    row = {
        "duracao": duration,
        "sample_rate": sr,
        "zcr": zcr,
        "rms": rms,
        "spectral_centroid": spectral_centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff
    }

    for i in range(13):
        row[f"mfcc_{i+1}"] = mfcc_means[i]

    return row


def augment_audio(audio, sr):
    augmented = []


    noise_factor = np.random.uniform(0.001, 0.003)
    noise = audio + noise_factor * np.random.randn(len(audio))
    augmented.append(noise)

    pitch = librosa.effects.pitch_shift(audio, sr=sr, n_steps=np.random.choice([-1, 1]))
    augmented.append(pitch)


    speed = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.9, 1.1))
    augmented.append(speed)

    return augmented



for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path) or label.lower() == "ruido":
        continue

    for file in os.listdir(label_path):
        file_path = os.path.join(label_path, file)

        try:
            audio, sr = librosa.load(file_path, sr=16000)

            augmented_versions = augment_audio(audio, sr)

   
            for aug_audio in augmented_versions[:MAX_AUG_PER_AUDIO]:
                features = extract_features(aug_audio, sr)

                row = {
                    "original_path": file_path,
                    "label": label,
                    "tipo": "augmentado"
                }

                row.update(features)
                data.append(row)

        except Exception as e:
            print(f"Erro em {file_path}: {e}")

df_noise = pd.DataFrame(data)
df_noise.to_csv(OUTPUT_CSV_NOISE, index=False)

print("✅ CSV de augmentation criado!")