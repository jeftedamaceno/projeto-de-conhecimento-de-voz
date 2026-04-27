import os
import numpy as np
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
from tensorflow.keras.models import load_model
import json
import keyboard
import time

SAMPLE_RATE = 16000
DURATION = 1.2
MODEL_PATH = "modelo_audio_da.h5"
LABELS_PATH = "labels_da.json"
# MODEL_PATH = "modelo_audio.h5"
# LABELS_PATH = "labels.json"


N_MELS = 128
MAX_LEN = 94

def gravar_audio(nome_arquivo):
    print("Gravando...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    print("Gravação finalizada!")

    audio = audio.flatten()
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    audio = np.clip(audio * 1.0, -1, 1)

    write(nome_arquivo, SAMPLE_RATE, audio.astype(np.float32))

    return nome_arquivo


def audio_para_mel(file_path):
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
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    # mel_db = librosa.power_to_db(mel, ref=np.max)
    # mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)
    if mel_db.shape[1] < MAX_LEN:
        pad = MAX_LEN - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad)))
    else:
        mel_db = mel_db[:, :MAX_LEN]

    return mel_db


model = load_model(MODEL_PATH)

with open(LABELS_PATH) as f:
    label_map = json.load(f)

inv_map = {v: k for k, v in label_map.items()}


def classificar_audio():
    caminho = gravar_audio("test_audios/CNN_teste.wav")

    mel = audio_para_mel(caminho)


    mel = np.expand_dims(mel, axis=-1)
    mel = np.expand_dims(mel, axis=0)

    pred = model.predict(mel)

    classe = inv_map[np.argmax(pred)]
    confianca = np.max(pred)

    print("\nResultado:")
    print("Classe prevista:", classe)
    print("Confiança:", confianca)


# classificar_audio()

def dirige_carro():

    while True:
        if keyboard.is_pressed('q'):
          
            break
        if keyboard.is_pressed('g'):
            classificar_audio()
        

dirige_carro()