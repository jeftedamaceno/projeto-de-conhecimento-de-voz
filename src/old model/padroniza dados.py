
import os
import librosa
import soundfile as sf
import numpy as np


origem = "dataset_vozes"
destino = "dataset"

classes = ["direita", "esquerda", "siga", "pare", "voltar"]
extensoes = (".wav", ".mp3", ".ogg", ".flac", ".m4a")

sample_rate = 22050
duracao_final = 1.2  
padding = 0.2       

for classe in classes:
    os.makedirs(os.path.join(destino, classe), exist_ok=True)

contador = {classe: 0 for classe in classes}


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


for aluno in os.listdir(origem):
    caminho_aluno = os.path.join(origem, aluno)

    if not os.path.isdir(caminho_aluno):
        continue

    for classe in classes:
        caminho_classe = os.path.join(caminho_aluno, classe)

        if not os.path.exists(caminho_classe):
            continue

        for arquivo in os.listdir(caminho_classe):

            if not arquivo.lower().endswith(extensoes):
                continue

            caminho_arquivo = os.path.join(caminho_classe, arquivo)

            try:
                y, sr = librosa.load(caminho_arquivo, sr=sample_rate)

                y = cortar_audio_inteligente(y, sr)

                novo_nome = f"{aluno}_{classe}_{contador[classe]}.wav"
                contador[classe] += 1

                destino_final = os.path.join(destino, classe, novo_nome)

                sf.write(destino_final, y, sample_rate)

            except Exception as e:
                print(f"Erro em {caminho_arquivo}: {e}")

print("Dataset pronto com corte inteligente!")