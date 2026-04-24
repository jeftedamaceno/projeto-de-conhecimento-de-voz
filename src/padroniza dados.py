# from pydub import AudioSegment
# from pydub.silence import detect_nonsilent
# from pathlib import Path
# from tqdm import tqdm

# # === CONFIG ===
# PASTA_ENTRADA = Path("dataset_vozes")
# PASTA_SAIDA = Path("dataset")

# PASTA_SAIDA.mkdir(exist_ok=True)

# EXTENSOES_VALIDAS = [".wav", ".mp3", ".ogg", ".flac", ".m4a"]

# PADDING = 300
# MIN_SILENCE_LEN = 400

# # === LOGS ===
# arquivos_com_erro = []
# arquivos_ignorados = []
# total_processados = 0


# def processar_audio(input_path, output_path):
#     global total_processados

#     try:
#         print(f"🎧 {input_path.name}")

#         # Ignorar arquivos muito pequenos
#         if input_path.stat().st_size < 1000:
#             arquivos_ignorados.append(str(input_path))
#             print(f"⚠️ Muito pequeno, ignorado")
#             return

#         audio = AudioSegment.from_file(input_path)

#         # Limitar duração (evita travamento)
#         if len(audio) > 10000:
#             audio = audio[:10000]

#         # Padronização
#         audio = audio.set_frame_rate(16000)
#         audio = audio.set_channels(1)

#         silence_thresh = audio.dBFS - 14

#         non_silent_ranges = detect_nonsilent(
#             audio,
#             min_silence_len=MIN_SILENCE_LEN,
#             silence_thresh=silence_thresh
#         )

#         # fallback → salva inteiro
#         if not non_silent_ranges:
#             print(f"⚠️ Sem detecção → salvo completo")
#             audio.export(output_path, format="wav")
#             total_processados += 1
#             return

#         chunks = []
#         for start, end in non_silent_ranges:
#             start = max(0, start - PADDING)
#             end = min(len(audio), end + PADDING)
#             chunks.append(audio[start:end])

#         output_audio = sum(chunks)
#         output_audio.export(output_path, format="wav")

#         total_processados += 1
#         print(f"✅ OK")

#     except Exception as e:
#         arquivos_com_erro.append((str(input_path), str(e)))
#         print(f"❌ ERRO: {input_path.name}")


# # === CONTADOR GLOBAL POR CLASSE ===
# contadores = {}

# # === LISTA TOTAL DE ARQUIVOS ===
# todos_arquivos = []

# for aluno in PASTA_ENTRADA.iterdir():
#     if aluno.is_dir():
#         for acao in aluno.iterdir():
#             if acao.is_dir():
#                 for arquivo in acao.iterdir():
#                     if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_VALIDAS:
#                         todos_arquivos.append((aluno, acao, arquivo))

# # === PROCESSAMENTO COM BARRA ===
# for aluno, acao, arquivo in tqdm(todos_arquivos, desc="Processando áudios"):
    
#     nome_acao = acao.name

#     pasta_saida = PASTA_SAIDA / nome_acao
#     pasta_saida.mkdir(exist_ok=True)

#     if nome_acao not in contadores:
#         contadores[nome_acao] = 1

#     numero = contadores[nome_acao]
#     novo_nome = f"{nome_acao}_{numero}.wav"
#     output_file = pasta_saida / novo_nome

#     processar_audio(arquivo, output_file)

#     contadores[nome_acao] += 1


# # === RELATÓRIO FINAL ===
# print("\n===== RELATÓRIO FINAL =====")
# print(f"✅ Processados: {total_processados}")
# print(f"⚠️ Ignorados (muito pequenos): {len(arquivos_ignorados)}")
# print(f"❌ Erros: {len(arquivos_com_erro)}")

# if arquivos_ignorados:
#     print("\nArquivos ignorados:")
#     for arq in arquivos_ignorados:
#         print(arq)

# if arquivos_com_erro:
#     print("\nArquivos com erro:")
#     for arq, erro in arquivos_com_erro:
#         print(f"{arq} → {erro}")
import os
import librosa
import soundfile as sf
import numpy as np

# Caminhos
origem = "dataset_vozes"
destino = "dataset"

classes = ["direita", "esquerda", "siga", "pare", "voltar"]
extensoes = (".wav", ".mp3", ".ogg", ".flac", ".m4a")

# Configurações
sample_rate = 22050
duracao_final = 1.2  # segundos (um pouco maior pra garantir a palavra)
padding = 0.2       # margem antes/depois da fala

# Criar pastas
for classe in classes:
    os.makedirs(os.path.join(destino, classe), exist_ok=True)

contador = {classe: 0 for classe in classes}


def cortar_audio_inteligente(y, sr):
    # Detectar partes com voz
    intervals = librosa.effects.split(y, top_db=20)

    if len(intervals) == 0:
        # fallback → áudio inteiro
        return y[:int(duracao_final * sr)]

    # Escolher o maior segmento (provável palavra)
    melhor = max(intervals, key=lambda x: x[1] - x[0])

    inicio, fim = melhor

    # Adicionar margem (padding)
    pad = int(padding * sr)
    inicio = max(0, inicio - pad)
    fim = min(len(y), fim + pad)

    y_cortado = y[inicio:fim]

    # Ajustar tamanho final
    tamanho = int(duracao_final * sr)

    if len(y_cortado) > tamanho:
        # corta do centro
        centro = len(y_cortado) // 2
        inicio = max(0, centro - tamanho // 2)
        y_cortado = y_cortado[inicio:inicio + tamanho]
    else:
        # padding com silêncio
        y_cortado = np.pad(y_cortado, (0, tamanho - len(y_cortado)))

    return y_cortado


# Loop principal
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