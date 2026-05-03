import numpy as np
import sounddevice as sd
from tensorflow.keras.models import load_model
import json
import keyboard

from utils import preprocess_audio, stft_manual, mel_filterbank

SAMPLE_RATE = 16000
DURATION = 1.2
TARGET_LENGTH = int(SAMPLE_RATE * DURATION)

MAX_LEN = 94

MODEL_PATH = "modelo_mic_simulacao1.h5"
LABELS_PATH = "labels_mic_simulacao1.json"

THRESHOLD = 0.5
ENTROPY_LIMIT = 1.4


def gravar_audio():

    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype='float32')

    sd.wait()
    return audio.flatten()


def audio_para_mel(audio):

    audio = preprocess_audio(audio, TARGET_LENGTH, training=False)

    spec = stft_manual(audio)
    spec_power = np.abs(spec) ** 2

    mel_fb = mel_filterbank(SAMPLE_RATE, 512, 128)

    mel_spec = np.dot(mel_fb, spec_power.T)

    mel_db = 10 * np.log10(mel_spec + 1e-10)

    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    if mel_db.shape[1] < MAX_LEN:
        mel_db = np.pad(mel_db, ((0,0),(0,MAX_LEN - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :MAX_LEN]

    return mel_db


def calcular_entropia(probs):
    probs = probs + 1e-10
    return -np.sum(probs * np.log(probs))


model = load_model(MODEL_PATH)

with open(LABELS_PATH) as f:
    label_map = json.load(f)

inv_map = {v: k for k, v in label_map.items()}

def classificar():
    print("gravando")
    audio = gravar_audio()

    mel = audio_para_mel(audio)

    mel = np.expand_dims(mel, axis=-1)
    mel = np.expand_dims(mel, axis=0)

    pred = model.predict(mel, verbose=0)[0]

    idx = np.argmax(pred)
    conf = pred[idx]
    ent = calcular_entropia(pred)

    classe = inv_map[idx]

    if classe == "ruido" or conf < THRESHOLD or ent > ENTROPY_LIMIT:
        classe = "desconhecido"

    print("\nClasse:", classe)
    print("Confiança:", conf)
    print("Entropia:", ent)

def loop():
    print("G = gravar | Q = sair")

    while True:
        if keyboard.is_pressed('q'):
            break
        if keyboard.is_pressed('g'):
            classificar()


loop()