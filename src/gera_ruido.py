import os
import numpy as np
import soundfile as sf
import wave

from utils import (
    normalizar,
    pad_or_trim,
    time_stretch,
    muda_pitch as pitch_shift
)

INPUT_DIR = "dataset_final"
OUTPUT_DIR = "dataset_ruido"

SAMPLE_RATE = 16000
TARGET_DURATION = 1.2
TARGET_LENGTH = int(SAMPLE_RATE * TARGET_DURATION)

MAX_POR_CLASSE = 800


def ruido_branco(audio, ruido=0.01):
    noise = np.random.normal(0, ruido, size=audio.shape)
    return audio + noise


def carrega_wav(path, mono=True):
    with wave.open(path, 'rb') as w:
        sr = w.getframerate()
        n_frames = w.getnframes()
        n_channels = w.getnchannels()
        amostra = w.getsampwidth()
        frames = w.readframes(n_frames)

    if amostra == 2:
        dtype = np.int16
    elif amostra == 4:
        dtype = np.int32
    else:
        raise ValueError("Formato não suportado")

    audio = np.frombuffer(frames, dtype=dtype)

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)
        if mono:
            audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    return audio, sr


def resample(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio

    duration = len(audio) / orig_sr
    new_length = int(duration * target_sr)

    old_indices = np.arange(len(audio))
    new_indices = np.linspace(0, len(audio) - 1, new_length)

    return np.interp(new_indices, old_indices, audio)


def carrega_audio(path, sr=None, mono=True):
    audio, orig_sr = carrega_wav(path, mono=mono)

    if sr is not None:
        audio = resample(audio, orig_sr, sr)
        return audio, sr

    return audio, orig_sr


def augment_audio(audio, sr):
    audios = []

    audio = pad_or_trim(audio, TARGET_LENGTH)

    audios.append(audio)

    audios.append(pad_or_trim(ruido_branco(audio, 0.003), TARGET_LENGTH))

    audios.append(pad_or_trim(ruido_branco(audio, 0.008), TARGET_LENGTH))

    audios.append(pad_or_trim(pitch_shift(audio, n_steps=1), TARGET_LENGTH))

    audios.append(pad_or_trim(time_stretch(audio, 0.95), TARGET_LENGTH))

    return audios


os.makedirs(OUTPUT_DIR, exist_ok=True)


for label in os.listdir(INPUT_DIR):
    input_label_path = os.path.join(INPUT_DIR, label)

    if not os.path.isdir(input_label_path):
        continue

    output_label_path = os.path.join(OUTPUT_DIR, label)
    os.makedirs(output_label_path, exist_ok=True)

    files = os.listdir(input_label_path)
    np.random.shuffle(files)

    contador = 0

    for file in files:
        if contador >= MAX_POR_CLASSE:
            break

        file_path = os.path.join(input_label_path, file)

        try:
            audio, sr = carrega_audio(file_path, sr=SAMPLE_RATE)

            audio = normalizar(audio)
            audio = pad_or_trim(audio, TARGET_LENGTH)

            versoes = augment_audio(audio, sr)

            for aug_audio in versoes:
                if contador >= MAX_POR_CLASSE:
                    break

                aug_audio = normalizar(aug_audio)
                aug_audio = pad_or_trim(aug_audio, TARGET_LENGTH)

                nome = f"{label}_{contador}.wav"
                caminho_saida = os.path.join(output_label_path, nome)

                sf.write(caminho_saida, aug_audio, SAMPLE_RATE)

                contador += 1

        except Exception as e:
            print(f"Erro em {file_path}: {e}")

    print(f"{label}: {contador} arquivos gerados")

print("Dataset com ruído criado!")