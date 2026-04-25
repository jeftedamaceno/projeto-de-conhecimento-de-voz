import os
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
from sklearn.ensemble import RandomForestClassifier

# ==============================
# CONFIG
# ==============================
SAMPLE_RATE = 16000
DURATION = 1.2  # segundos
TEST_FOLDER = "test_audios"
MODEL_PATH = "dataset.csv"

os.makedirs(TEST_FOLDER, exist_ok=True)

# ==============================
# 1. FUNÇÃO PARA GRAVAR ÁUDIO
# ==============================
def gravar_audio(nome_arquivo):
    print("Gravando...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    print("Gravação finalizada!")

    caminho = os.path.join(TEST_FOLDER, nome_arquivo)
    write(caminho, SAMPLE_RATE, audio)

    return caminho

# ==============================
# 2. EXTRAIR FEATURES (igual dataset)
# ==============================
def extrair_features(file_path):
    # audio, sr = librosa.load(file_path, sr=16000)
    audio, sr = librosa.load(file_path, sr=16000)

    # 🔥 NORMALIZAÇÃO
    audio = normalizar_audio(audio)
    audio = audio * 6
    audio = np.clip(audio, -1, 1)

    duration = librosa.get_duration(y=audio, sr=sr)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    rms = np.mean(librosa.feature.rms(y=audio))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc.T, axis=0)

    features = [
        duration,
        sr,
        zcr,
        rms,
        spectral_centroid
    ]

    features.extend(mfcc_means)

    return np.array(features).reshape(1, -1)

# ==============================
# 3. TREINAR MODELO (a partir do CSV)
# ==============================
def treinar_modelo():
    df = pd.read_csv(MODEL_PATH)

    df = df.drop(columns=["caminho"])

    X = df.drop(columns=["label"])
    y = df["label"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X, y)

    return model, X.columns

# ==============================
# 4. PIPELINE COMPLETO
# ==============================
def classificar_audio():
    # grava
    caminho_audio = gravar_audio("teste.wav")

    # extrai features
    features = extrair_features(caminho_audio)

    # treina modelo
    model, colunas = treinar_modelo()

    # transforma em DataFrame (garante ordem correta)
    features_df = pd.DataFrame(features, columns=colunas)

    # previsão
    pred = model.predict(features_df)
    prob = model.predict_proba(features_df)

    print("\n🔮 Resultado:")
    print("Classe prevista:", pred[0])
    print("Confiança:", np.max(prob))

# ==============================
# EXECUTAR
# ==============================
def normalizar_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio
classificar_audio()