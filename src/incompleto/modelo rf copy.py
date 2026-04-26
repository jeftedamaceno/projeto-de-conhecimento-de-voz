import os
import numpy as np
import pandas as pd
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==============================
# CONFIG
# ==============================
DATASET_PATH = "dataset"
MODEL_REAL_PATH = "modelo_sem_aug.pkl"
MODEL_AUG_PATH = "modelo_com_aug.pkl"
COLS_PATH = "colunas.pkl"

SAMPLE_RATE = 16000
TARGET_DURATION = 1.2

# ==============================
# PADRONIZAÇÃO
# ==============================
def padronizar_audio(audio, sr):
    target_len = int(TARGET_DURATION * sr)

    if len(audio) > target_len:
        audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)))

    return audio

# ==============================
# EXTRAÇÃO DE FEATURES
# ==============================
def extrair_features(audio, sr):
    feats = {}

    # normalização
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    feats["zcr"] = np.mean(librosa.feature.zero_crossing_rate(audio))
    feats["rms"] = np.mean(librosa.feature.rms(y=audio))
    feats["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    feats["bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    feats["rolloff"] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    feats["contrast"] = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr))
    feats["energy"] = np.sum(audio**2)

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    feats["chroma"] = np.mean(chroma)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    for i, val in enumerate(np.mean(mfcc.T, axis=0)):
        feats[f"mfcc_{i}"] = val

    return feats

# ==============================
# DATA AUGMENTATION
# ==============================
def augmentar_audio(audio, sr):
    audios = []

    noise = np.random.randn(len(audio))
    audios.append(audio + 0.005 * noise)

    audios.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=2))
    audios.append(librosa.effects.time_stretch(audio, rate=0.9))

    return audios

# ==============================
# CRIAR DATASET
# ==============================
def criar_dataset():
    data = []

    for label in os.listdir(DATASET_PATH):
        path_label = os.path.join(DATASET_PATH, label)

        if not os.path.isdir(path_label):
            continue

        for file in os.listdir(path_label):
            file_path = os.path.join(path_label, file)

            try:
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                audio = padronizar_audio(audio, sr)

                # ORIGINAL
                feats = extrair_features(audio, sr)
                feats["label"] = label
                feats["is_augmented"] = 0
                data.append(feats)

                # AUGMENTATION
                for aug_audio in augmentar_audio(audio, sr):
                    aug_audio = padronizar_audio(aug_audio, sr)

                    feats_aug = extrair_features(aug_audio, sr)
                    feats_aug["label"] = label
                    feats_aug["is_augmented"] = 1
                    data.append(feats_aug)

            except Exception as e:
                print(f"Erro em {file_path}: {e}")

    return pd.DataFrame(data)

# ==============================
# TREINAR MODELOS
# ==============================
def treinar():
    df = criar_dataset()

    print("Dataset criado:", df.shape)

    # dados reais
    df_real = df[df["is_augmented"] == 0]

    X_real = df_real.drop(columns=["label", "is_augmented"])
    y_real = df_real["label"]

    X_all = df.drop(columns=["label", "is_augmented"])
    y_all = df["label"]

    # split SOMENTE REAL
    X_train, X_test, y_train, y_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42
    )

    # modelo sem augmentation
    model_real = RandomForestClassifier(n_estimators=300, max_depth=15)
    model_real.fit(X_train, y_train)

    # modelo com augmentation
    model_aug = RandomForestClassifier(n_estimators=300, max_depth=15)
    model_aug.fit(X_all, y_all)

    # salvar
    joblib.dump(model_real, MODEL_REAL_PATH)
    joblib.dump(model_aug, MODEL_AUG_PATH)
    joblib.dump(X_train.columns.tolist(), COLS_PATH)

    print("✅ Modelos salvos!")

# ==============================
# EXECUTAR
# ==============================

treinar()