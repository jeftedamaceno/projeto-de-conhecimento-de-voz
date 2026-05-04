# import numpy as np
# import sounddevice as sd
# from tensorflow.keras.models import load_model
# import json

# from utils import stft_manual, mel_filterbank, preprocess_audio


# SAMPLE_RATE = 16000
# FRAME_SIZE = 1024
# HOP = 512

# TARGET_LENGTH = int(1.2 * SAMPLE_RATE)

# MODEL_PATH = "modelo_mic_simulacao1.h5"
# LABELS_PATH = "labels_mic_simulacao1.json"

# THRESHOLD_ENERGY = 0.01
# SILENCE_FRAMES = 10  

# model = load_model(MODEL_PATH)

# with open(LABELS_PATH) as f:
#     label_map = json.load(f)

# inv_map = {v: k for k, v in label_map.items()}


# def audio_para_mel(audio):

#     audio = preprocess_audio(audio, TARGET_LENGTH, training=False)

#     spec = stft_manual(audio, n_fft=512, hop_length=160)
#     spec_power = np.abs(spec) ** 2

#     mel_fb = mel_filterbank(16000, 512, 128)

#     mel_spec = np.dot(mel_fb, spec_power.T)

#     mel_db = 10 * np.log10(mel_spec + 1e-10)

#     mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

#     if mel_db.shape[1] < 94:
#         mel_db = np.pad(mel_db, ((0,0),(0,94 - mel_db.shape[1])))
#     else:
#         mel_db = mel_db[:, :94]

#     return mel_db


# def classificar(audio):

#     mel = audio_para_mel(audio)

#     mel = np.expand_dims(mel, axis=-1)
#     mel = np.expand_dims(mel, axis=0)

#     pred = model.predict(mel, verbose=0)[0]

#     classe = inv_map[np.argmax(pred)]
#     confianca = np.max(pred)

#     print(f"\n {classe} ({confianca:.2f})")


# def loop_realtime():

#     print("🎤 Escutando... (Ctrl+C para sair)")

#     buffer = []
#     gravando = False
#     silencio_contador = 0

#     def callback(indata, frames, time, status):
#         nonlocal buffer, gravando, silencio_contador

#         audio = indata[:, 0]

#         energia = np.mean(np.abs(audio))

#         # INÍCIO DA FALA
#         if energia > THRESHOLD_ENERGY:
#             buffer.extend(audio)
#             gravando = True
#             silencio_contador = 0

#         elif gravando:
#             buffer.extend(audio)
#             silencio_contador += 1

#             # FIM DA FALA
#             if silencio_contador > SILENCE_FRAMES:

#                 if len(buffer) > SAMPLE_RATE * 0.5:
#                     audio_final = np.array(buffer)

#                     classificar(audio_final)

#                 buffer = []
#                 gravando = False
#                 silencio_contador = 0

#     with sd.InputStream(
#         samplerate=SAMPLE_RATE,
#         channels=1,
#         blocksize=FRAME_SIZE,
#         callback=callback
#     ):
#         while True:
#             pass

# # =========================
# # RUN
# # =========================
# loop_realtime()
import numpy as np
import sounddevice as sd
import queue
import time
import tensorflow as tf

from utils import preprocess_audio, stft_manual, mel_filterbank

# =========================
# CONFIG
# =========================
SAMPLE_RATE = 16000
DURATION = 1.2
TARGET_LENGTH = int(SAMPLE_RATE * DURATION)

N_MELS = 128
MAX_LEN = 94

WINDOW_SIZE = int(0.5 * SAMPLE_RATE)   # 0.5s
STRIDE = int(0.2 * SAMPLE_RATE)        # 0.2s

THRESHOLD = 0.75
ENERGY_THRESHOLD = 0.01

# =========================
# CARREGAR MODELO
# =========================
model = tf.keras.models.load_model("modelo_mic_simulacao1.h5")

# =========================
# FILA DE ÁUDIO
# =========================
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

# =========================
# DETECÇÃO DE VOZ (VAD)
# =========================
def detectar_voz(audio):
    energia = np.mean(audio ** 2)
    return energia > ENERGY_THRESHOLD

# =========================
# CONVERSÃO PARA MEL
# =========================
def audio_para_mel(audio):

    audio = preprocess_audio(audio, TARGET_LENGTH, training=False)

    spec = stft_manual(audio)
    spec_power = np.abs(spec) ** 2

    mel_fb = mel_filterbank(SAMPLE_RATE, 512, N_MELS)
    mel_spec = np.dot(mel_fb, spec_power.T)

    mel_db = 10 * np.log10(mel_spec + 1e-10)

    # normalização
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    # padding / corte
    if mel_db.shape[1] < MAX_LEN:
        mel_db = np.pad(mel_db, ((0,0),(0,MAX_LEN - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :MAX_LEN]

    return mel_db

# =========================
# PREDIÇÃO
# =========================
def prever(audio):

    mel = audio_para_mel(audio)
    mel = np.expand_dims(mel, axis=(0, -1))

    probs = model.predict(mel, verbose=0)[0]

    classe = np.argmax(probs)
    confianca = np.max(probs)

    if confianca < THRESHOLD:
        return "desconhecido", confianca

    return classe, confianca

# =========================
# LOOP PRINCIPAL
# =========================
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

                # processar quando tiver janela suficiente
                if len(buffer) >= WINDOW_SIZE:

                    janela = buffer[:WINDOW_SIZE]
                    buffer = buffer[STRIDE:]  # sliding window

                    # VAD
                    if not detectar_voz(janela):
                        continue

                    classe, conf = prever(janela)

                    print(f"🎯 {classe} | Confiança: {conf:.2f}")

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n Encerrado")


main()