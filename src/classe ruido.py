import os
import librosa
import soundfile as sf
import numpy as np

# Caminhos
origem = "ruido"
destino = os.path.join("ruido tratado", "ruido")

extensoes = (".wav", ".mp3", ".ogg", ".flac", ".m4a")

# Configurações
sample_rate = 22050
duracao_final = 1.2
padding = 0.2

# Criar pasta destino
os.makedirs(destino, exist_ok=True)

contador = 0


def cortar_audio_inteligente(y, sr):
    intervals = librosa.effects.split(y, top_db=20)

    if len(intervals) == 0:
        return y[:int(duracao_final * sr)]

    melhor = max(intervals, key=lambda x: x[1] - x[0])

    inicio, fim = melhor

    pad = int(padding * sr)
    inicio = max(0, inicio - pad)
    fim = min(len(y), fim + pad)

    y_cortado = y[inicio:fim]

    tamanho = int(duracao_final * sr)

    if len(y_cortado) > tamanho:
        centro = len(y_cortado) // 2
        inicio = max(0, centro - tamanho // 2)
        y_cortado = y_cortado[inicio:inicio + tamanho]
    else:
        y_cortado = np.pad(y_cortado, (0, tamanho - len(y_cortado)))

    return y_cortado


# Loop principal (sem alunos e sem classes)
for arquivo in os.listdir(origem):

    if not arquivo.lower().endswith(extensoes):
        continue

    caminho_arquivo = os.path.join(origem, arquivo)

    try:
        y, sr = librosa.load(caminho_arquivo, sr=sample_rate)

        y = cortar_audio_inteligente(y, sr)

        novo_nome = f"ruido_{contador}.wav"
        contador += 1

        destino_final = os.path.join(destino, novo_nome)

        sf.write(destino_final, y, sample_rate)

    except Exception as e:
        print(f"Erro em {caminho_arquivo}: {e}")

print("Dataset de ruído pronto!")