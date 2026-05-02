import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write, read
from tensorflow.keras.models import load_model
import json
import keyboard

from utils import stft_manual, mel_filterbank, normalizar, gerar_log_mel_spectrogram
# =========================
# CONFIG
# =========================
SAMPLE_RATE = 16000
DURATION = 1.2

MODEL_PATH = "test_beta_model.h5"
LABELS_PATH = "test_beta_model_labels.json"

N_MELS = 128
MAX_LEN = 94

# controle de erro
THRESHOLD = 0.75
ENTROPY_LIMIT = 1.8


# =========================
# GRAVAÇÃO
# =========================
def gravar_audio(nome_arquivo):
    print("Gravando...")

    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1)
    sd.wait()

    audio = audio.flatten()

    # normalização
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    write(nome_arquivo, SAMPLE_RATE, audio.astype(np.float32))

    print("Gravação finalizada!")
    return nome_arquivo


# =========================
# ENTROPIA
# =========================
def calcular_entropia(probs):
    probs = probs + 1e-10
    return -np.sum(probs * np.log(probs))


# =========================
# ÁUDIO -> MEL (MANUAL)
# =========================
def audio_para_mel(file_path):

    sr, audio = read(file_path)
    audio = audio.astype(np.float32)

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    audio = normalizar(audio)

    # ===== IGUAL AO TREINO =====
    spec = stft_manual(audio, n_fft=512, hop_length=160)
    spec_power = np.abs(spec) ** 2

    mel_fb = mel_filterbank(sr=16000, n_fft=512, n_mels=128)

    mel_spec = np.dot(mel_fb, spec_power.T)

    mel_db = 10 * np.log10(mel_spec + 1e-10)

    # normalização igual
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

    # padding igual
    if mel_db.shape[1] < 94:
        pad = 94 - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad)))
    else:
        mel_db = mel_db[:, :94]

    return mel_db


# =========================
# LOAD
# =========================
model = load_model(MODEL_PATH)

with open(LABELS_PATH) as f:
    label_map = json.load(f)

inv_map = {v: k for k, v in label_map.items()}


# =========================
# CLASSIFICAÇÃO
# =========================
def classificar_audio():

    caminho = gravar_audio("test_audios/teste.wav")

    mel = audio_para_mel(caminho)

    mel = np.expand_dims(mel, axis=-1)
    mel = np.expand_dims(mel, axis=0)

    pred = model.predict(mel, verbose=0)[0]

    classe_idx = np.argmax(pred)
    confianca = pred[classe_idx]
    classe = inv_map[classe_idx]

    entropia = calcular_entropia(pred)
    print("\nProbabilidades completas:")
    

    # ===== REGRAS INTELIGENTES =====
    if confianca < THRESHOLD or entropia > ENTROPY_LIMIT:
        classe = "ruido"

    print("\nResultado:")
    print("Classe:", classe)
    print("Confiança:", confianca)
    print("Entropia:", entropia)
    for i, p in enumerate(pred):
        print(f"{inv_map[i]}: {p:.4f}")


# =========================
# LOOP
# =========================
def loop():

    print("Pressione G para gravar | Q para sair")

    while True:
        if keyboard.is_pressed('q'):
            break

        if keyboard.is_pressed('g'):
            classificar_audio()


loop()

