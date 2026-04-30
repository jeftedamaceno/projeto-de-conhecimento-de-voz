import os
import numpy as np
import soundfile as sf
import subprocess

from utils import normalizar

# ORIGEM = "dataset_vozes_old"
# DESTINO = "dataset_final"
BASE_DIR = r"projeto de reconhecimento de voz AMRP"

ORIGEM = os.path.join(BASE_DIR, "dataset_vozes_old")
DESTINO = os.path.join(BASE_DIR, "dataset_final")

CLASSES = ["direita", "esquerda", "siga", "pare", "voltar"]
EXTENSOES = (".wav", ".flac", ".ogg", ".m4a", ".mp3")

SAMPLE_RATE = 16000
DURACAO = 1.2
PADDING = 0.2

FFMPEG_PATH = r"ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"
import subprocess

subprocess.run([FFMPEG_PATH, "-version"])

def converter_audio(entrada):
    saida = entrada + "_temp.wav"

    comando = [
        FFMPEG_PATH,
        "-y",
        "-i", entrada,
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-loglevel", "error",
        saida
    ]

    result = subprocess.run(comando, capture_output=True, text=True)

    if result.returncode != 0:
        print("Erro FFmpeg:", result.stderr)
        return None

    return saida


def energia(signal):
    return np.convolve(signal**2, np.ones(400)/400, mode='same')


def ajustar_tamanho(audio, sr):
    tamanho = int(DURACAO * sr)

    if len(audio) > tamanho:
        centro = len(audio) // 2
        inicio = centro - tamanho // 2
        return audio[inicio:inicio + tamanho]
    else:
        return np.pad(audio, (0, tamanho - len(audio)))


def cortar_audio(audio, sr):
    env = energia(audio)
    threshold = np.max(env) * 0.1

    indices = np.where(env > threshold)[0]

    if len(indices) == 0:
        return ajustar_tamanho(audio, sr)

    inicio = max(0, indices[0] - int(PADDING * sr))
    fim = min(len(audio), indices[-1] + int(PADDING * sr))

    audio = audio[inicio:fim]
    return ajustar_tamanho(audio, sr)


def reamostrar(audio, sr_origem, sr_destino):
    if sr_origem == sr_destino:
        return audio

    duracao = len(audio) / sr_origem
    novo_tamanho = int(duracao * sr_destino)

    indices = np.linspace(0, len(audio) - 1, novo_tamanho)

    indices_int = np.floor(indices).astype(int)
    frac = indices - indices_int

    indices_int2 = np.clip(indices_int + 1, 0, len(audio) - 1)

    return (1 - frac) * audio[indices_int] + frac * audio[indices_int2]


os.makedirs(DESTINO, exist_ok=True)
for classe in CLASSES:
    os.makedirs(os.path.join(DESTINO, classe), exist_ok=True)

contador = {classe: 0 for classe in CLASSES}


for aluno in os.listdir(ORIGEM):
    caminho_aluno = os.path.join(ORIGEM, aluno)

    if not os.path.isdir(caminho_aluno):
        continue

    for classe in CLASSES:
        caminho_classe = os.path.join(caminho_aluno, classe)

        if not os.path.exists(caminho_classe):
            continue

        for arquivo in os.listdir(caminho_classe):

            if not arquivo.lower().endswith(EXTENSOES):
                continue

            caminho_arquivo = os.path.join(caminho_classe, arquivo)
            temp_file = None

            try:
                if not arquivo.lower().endswith(".wav"):
                    temp_file = converter_audio(caminho_arquivo)

                    if temp_file is None:
                        continue

                    caminho_arquivo = temp_file

                audio, sr = sf.read(caminho_arquivo)

                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)

                audio = reamostrar(audio, sr, SAMPLE_RATE)

                audio = normalizar(audio)

                audio = cortar_audio(audio, SAMPLE_RATE)

                nome = f"{classe}_{contador[classe]}.wav"
                contador[classe] += 1
                print("Processando:", caminho_arquivo)

                if not os.path.exists(caminho_arquivo):
                    print("❌ NÃO EXISTE:", caminho_arquivo)
                    continue

                destino_final = os.path.join(DESTINO, classe, nome)
                sf.write(destino_final, audio, SAMPLE_RATE)

            except Exception as e:
                print(f"Erro em {caminho_arquivo}: {e}")

            finally:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)


print("dataset_final criado!")