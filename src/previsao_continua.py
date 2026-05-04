
import numpy as np
import sounddevice as sd
import queue
import time
import tensorflow as tf
import json
from utils import preprocess_audio, stft_manual, mel_filterbank


SAMPLE_RATE = 16000
DURATION = 1.2
TARGET_LENGTH = int(SAMPLE_RATE * DURATION)

N_MELS = 128
MAX_LEN = 94

WINDOW_SIZE = int(0.5 * SAMPLE_RATE)  
STRIDE = int(0.2 * SAMPLE_RATE)        

THRESHOLD = 0.72
ENERGY_THRESHOLD = 0.01


model = tf.keras.models.load_model("modelo_mic_simulacao1.h5")

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

    

with open("labels_mic_simulacao1.json") as f:
    label_map = json.load(f)

inv_map = {v: k for k, v in label_map.items()}
def detectar_voz(audio):
    audio = audio / (np.max(np.abs(audio)) + 1e-4)
    energia = np.mean(audio ** 2)
    return energia > 0.01


def audio_para_mel(audio):

    audio = preprocess_audio(audio, TARGET_LENGTH, training=False)

    spec = stft_manual(audio, n_fft=512, hop_length=160)
    spec_power = np.abs(spec) ** 2

    mel_fb = mel_filterbank(SAMPLE_RATE, 512, N_MELS)
    mel_spec = np.dot(mel_fb, spec_power.T)

    mel_db = 10 * np.log10(mel_spec + 1e-10)

    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    if mel_db.shape[1] < MAX_LEN:
        mel_db = np.pad(mel_db, ((0,0),(0,MAX_LEN - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :MAX_LEN]

    return mel_db


def prever(audio):

    mel = audio_para_mel(audio)
    mel = np.expand_dims(mel, axis=(0, -1))

    probs = model.predict(mel, verbose=0)[0]

    classe_idx = np.argmax(probs)
    confianca = np.max(probs)

    classe = inv_map[classe_idx]

    if confianca < THRESHOLD:
        return "desconhecido", confianca

    return classe, confianca
def main():

    print(" Gravando... pressione CTRL+C para parar")

    buffer = np.zeros(0)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback
    ):

        try:
            while True:
                

                if not audio_queue.empty():
                    data = audio_queue.get().flatten()
                    buffer = np.concatenate((buffer, data))

              
                if len(buffer) >= WINDOW_SIZE:

                    janela = buffer[:WINDOW_SIZE]
                    buffer = buffer[STRIDE:] 

               
                    if not detectar_voz(janela):
                        continue

                    classe, conf = prever(janela)

                    print(f" {classe} | Confiança: {conf:.2f}")

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n Encerrado")


main()