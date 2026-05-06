import numpy as np
import sounddevice as sd
import queue
import time
import threading
import tensorflow as tf
import json
import pygame
import math

from utils import preprocess_audio, stft_manual, mel_filterbank


SAMPLE_RATE = 16000
TARGET_LENGTH = int(1.2 * SAMPLE_RATE)

WINDOW_SIZE = int(0.5 * SAMPLE_RATE)
STRIDE = int(0.2 * SAMPLE_RATE)

THRESHOLD = 0.72
ENERGY_THRESHOLD = 0.01


HIST_SIZE = 5
COOLDOWN = 0.8  

model = tf.keras.models.load_model("modelo_mic_simulacao1.h5")

with open("labels_mic_simulacao1.json") as f:
    label_map = json.load(f)

inv_map = {v: k for k, v in label_map.items()}


audio_queue = queue.Queue()


current_command = "stop"
historico = []
last_command_time = 0


MAPA_COMANDOS = {
    "pare": "stop",
    "frente": "straight",
    "esquerda": "left",
    "direita": "right"
}


def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())


def detectar_voz(audio):
    audio = audio / (np.max(np.abs(audio)) + 1e-4)
    energia = np.mean(audio ** 2)
    return energia > ENERGY_THRESHOLD


def audio_para_mel(audio):

    audio = preprocess_audio(audio, TARGET_LENGTH, training=False)

    spec = stft_manual(audio, n_fft=512, hop_length=160)
    spec_power = np.abs(spec) ** 2

    mel_fb = mel_filterbank(SAMPLE_RATE, 512, 128)
    mel_spec = np.dot(mel_fb, spec_power.T)

    mel_db = 10 * np.log10(mel_spec + 1e-10)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    if mel_db.shape[1] < 94:
        mel_db = np.pad(mel_db, ((0,0),(0,94 - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :94]

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


def atualizar_comando(novo):

    global historico, current_command, last_command_time

    historico.append(novo)

    if len(historico) > HIST_SIZE:
        historico.pop(0)


    comando_final = max(set(historico), key=historico.count)

    agora = time.time()


    if (agora - last_command_time > COOLDOWN) and comando_final != current_command:

        current_command = MAPA_COMANDOS.get(comando_final, "stop")
        last_command_time = agora

        print(f" COMANDO FINAL: {current_command}")


def audio_thread():

    buffer = np.zeros(0)

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

            print(f"{classe} ({conf:.2f})")

            if classe != "desconhecido":
                atualizar_comando(classe)


def simulador():

    global current_command

    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()

    x, y, theta = 0, 0, 0

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

    
        cmd = current_command

        if cmd == "straight":
            x += 0.02
        elif cmd == "left":
            theta += 0.05
        elif cmd == "right":
            theta -= 0.05
        elif cmd == "stop":
            pass

        screen.fill((30,30,30))
        pygame.display.flip()
        clock.tick(60)


def main():

    print("Sistema iniciado...")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback
    )
    stream.start()

    threading.Thread(target=audio_thread, daemon=True).start()

    simulador()

main()