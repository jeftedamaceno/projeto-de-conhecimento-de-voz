import librosa
import numpy as np
import os
from tqdm import tqdm

INPUT_PATH = "dataset"
OUTPUT_PATH = "spectrograms"

SAMPLE_RATE = 16000
DURATION = 1  # segundos
SAMPLES = SAMPLE_RATE * DURATION

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ==============================
# PAD OU CORTE CENTRAL
# ==============================
def resize_feature(x, target_shape=(128, 128)):
# cortar ou pad altura
    if x.shape[0] > target_shape[0]:
        x = x[:target_shape[0], :]
    else:
        pad_h = target_shape[0] - x.shape[0]
        x = np.pad(x, ((0, pad_h), (0, 0)))

    # cortar ou pad largura
    if x.shape[1] > target_shape[1]:
        x = x[:, :target_shape[1]]
    else:
        pad_w = target_shape[1] - x.shape[1]
        x = np.pad(x, ((0, 0), (0, pad_w)))
    return x
def pad_or_trim(audio):
    if len(audio) > SAMPLES:
        start = (len(audio) - SAMPLES) // 2
        return audio[start:start + SAMPLES]
    else:
        pad = SAMPLES - len(audio)
        return np.pad(audio, (pad // 2, pad - pad // 2))


# ==============================
# EXTRAÇÃO DE FEATURES
# ==============================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # 🔥 PROTEÇÃO CONTRA DIVISÃO POR ZERO
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    y = pad_or_trim(y)

    # MEL
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel = librosa.power_to_db(mel)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # DELTA
    delta = librosa.feature.delta(mfcc)

    # Garantir tamanho fixo (128x128)
    # mel = mel[:, :128]
    # mfcc = mfcc[:, :128]
    # delta = delta[:, :128]

    mel = resize_feature(mel)
    mfcc = resize_feature(mfcc)
    delta = resize_feature(delta)

    

    # Padding se necessário
    def fix_shape(x):
        if x.shape[1] < 128:
            pad_width = 128 - x.shape[1]
            return np.pad(x, ((0, 0), (0, pad_width)))
        return x

    mel = fix_shape(mel)
    mfcc = fix_shape(mfcc)
    delta = fix_shape(delta)

    # Empilhar canais (tipo imagem RGB)
    img = np.stack([mel, mfcc, delta], axis=-1)

    return img


# ==============================
# PROCESSAMENTO DO DATASET
# ==============================
EXTENSOES_VALIDAS = (".wav", ".mp3", ".ogg", ".flac", ".m4a")

for classe in os.listdir(INPUT_PATH):
    class_path = os.path.join(INPUT_PATH, classe)

    if not os.path.isdir(class_path):
        continue

    out_class = os.path.join(OUTPUT_PATH, classe)
    os.makedirs(out_class, exist_ok=True)

    for file in tqdm(os.listdir(class_path), desc=classe):

        if not file.lower().endswith(EXTENSOES_VALIDAS):
            continue

        try:
            file_path = os.path.join(class_path, file)

            img = extract_features(file_path)

            # 🔥 CORREÇÃO DA EXTENSÃO (FUNCIONA PRA QUALQUER FORMATO)
            nome_base = os.path.splitext(file)[0]
            save_path = os.path.join(out_class, nome_base + ".npy")

            np.save(save_path, img)

        except Exception as e:
            print(f"Erro: {file} -> {e}")