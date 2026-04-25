# import librosa
# import numpy as np
# import matplotlib.pyplot as plt

# def gerar_spectrograma(audio_path):
#     y, sr = librosa.load(audio_path, sr=16000)

#     spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
#     spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

#     plt.figure(figsize=(2,2))
#     plt.axis('off')
#     plt.imshow(spectrogram_db)

#     temp_path = "temp.png"
#     plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
#     plt.close()

#     return temp_path

# import tensorflow as tf
# import json
# from tensorflow.keras.preprocessing import image

# # carregar modelo
# model = tf.keras.models.load_model("modelo_voz.h5")

# # carregar classes
# with open("classes.json") as f:
#     class_names = json.load(f)

# def prever(audio_path):
#     img_path = gerar_spectrograma(audio_path)

#     img = image.load_img(img_path, target_size=(128,128))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0

#     predictions = model.predict(img_array)
#     classe = class_names[np.argmax(predictions)]

#     print("Classe prevista:", classe)
#     print("Confiança:", np.max(predictions))

#     return classe

# prever("direita test.ogg")

import sounddevice as sd
import numpy as np
import queue
import librosa
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import time
import os

# ==============================
# CONFIG
# ==============================

SAMPLE_RATE = 16000
# THRESHOLD = 0.01        # sensibilidade da voz
# SILENCE_TIME = 1.0      # segundos de silêncio para parar gravação
THRESHOLD = 0.003
SILENCE_TIME = 1.2
q = queue.Queue()

# ==============================
# CARREGAR MODELO
# ==============================

model = tf.keras.models.load_model("modelo_voz.h5")

with open("classes.json") as f:
    class_names = json.load(f)

# ==============================
# CALLBACK DO MICROFONE
# ==============================

def audio_callback(indata, frames, time_info, status):
    volume = np.linalg.norm(indata) / frames
    q.put((indata.copy(), volume))

# ==============================
# GERAR ESPECTROGRAMA
# ==============================

def gerar_spectrograma(audio, sr=16000):
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(2,2))
    plt.axis('off')
    plt.imshow(spectrogram_db)

    temp_path = "temp.png"
    plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return temp_path

# ==============================
# PREVISÃO
# ==============================

def prever_audio(audio):
    temp_img = gerar_spectrograma(audio)

    img = tf.keras.preprocessing.image.load_img(temp_img, target_size=(128,128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred = model.predict(img_array)
    classe = class_names[np.argmax(pred)]
    confianca = np.max(pred)

    print(f"\n🎯 Classe: {classe} | Confiança: {confianca:.2f}")

    os.remove(temp_img)

# ==============================
# LOOP PRINCIPAL
# ==============================

def ouvir_microfone():
    print("🎤 Aguardando fala...")

    gravando = False
    audio_buffer = []
    ultimo_som = time.time()

    with sd.InputStream(callback=audio_callback,
                        channels=1,
                        samplerate=SAMPLE_RATE):

        while True:
            data, volume = q.get()

            # Detecta início da fala
            if volume > THRESHOLD:
                if not gravando:
                    print("🟢 Fala detectada!")
                    gravando = True
                    audio_buffer = []

                ultimo_som = time.time()
                audio_buffer.append(data)

            # Detecta silêncio
            elif gravando:
                audio_buffer.append(data)

                if time.time() - ultimo_som > SILENCE_TIME:
                    print("🔴 Fim da fala")

                    # juntar áudio
                    audio_np = np.concatenate(audio_buffer, axis=0).flatten()

                    # normalizar
                    audio_np = audio_np / np.max(np.abs(audio_np))

                    # prever
                    prever_audio(audio_np)

                    gravando = False
                    audio_buffer = []
    print("Volume:", volume)

# ==============================
# EXECUTAR
# ==============================

# ouvir_microfone()
print("Gravando 1.5 segundos...")
audio = sd.rec(int(1.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
sd.wait()

audio = audio.flatten()
audio = audio / np.max(np.abs(audio))

prever_audio(audio)