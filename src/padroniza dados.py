from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
PASTA_ENTRADA = Path("dataset_vozes")
PASTA_SAIDA = Path("dataset")

PASTA_SAIDA.mkdir(exist_ok=True)

EXTENSOES_VALIDAS = [".wav", ".mp3", ".ogg", ".flac", ".m4a"]

PADDING = 300
MIN_SILENCE_LEN = 400

# === LOGS ===
arquivos_com_erro = []
arquivos_ignorados = []
total_processados = 0


def processar_audio(input_path, output_path):
    global total_processados

    try:
        print(f"🎧 {input_path.name}")

        # Ignorar arquivos muito pequenos
        if input_path.stat().st_size < 1000:
            arquivos_ignorados.append(str(input_path))
            print(f"⚠️ Muito pequeno, ignorado")
            return

        audio = AudioSegment.from_file(input_path)

        # Limitar duração (evita travamento)
        if len(audio) > 10000:
            audio = audio[:10000]

        # Padronização
        audio = audio.set_frame_rate(16000)
        audio = audio.set_channels(1)

        silence_thresh = audio.dBFS - 14

        non_silent_ranges = detect_nonsilent(
            audio,
            min_silence_len=MIN_SILENCE_LEN,
            silence_thresh=silence_thresh
        )

        # fallback → salva inteiro
        if not non_silent_ranges:
            print(f"⚠️ Sem detecção → salvo completo")
            audio.export(output_path, format="wav")
            total_processados += 1
            return

        chunks = []
        for start, end in non_silent_ranges:
            start = max(0, start - PADDING)
            end = min(len(audio), end + PADDING)
            chunks.append(audio[start:end])

        output_audio = sum(chunks)
        output_audio.export(output_path, format="wav")

        total_processados += 1
        print(f"✅ OK")

    except Exception as e:
        arquivos_com_erro.append((str(input_path), str(e)))
        print(f"❌ ERRO: {input_path.name}")


# === CONTADOR GLOBAL POR CLASSE ===
contadores = {}

# === LISTA TOTAL DE ARQUIVOS ===
todos_arquivos = []

for aluno in PASTA_ENTRADA.iterdir():
    if aluno.is_dir():
        for acao in aluno.iterdir():
            if acao.is_dir():
                for arquivo in acao.iterdir():
                    if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_VALIDAS:
                        todos_arquivos.append((aluno, acao, arquivo))

# === PROCESSAMENTO COM BARRA ===
for aluno, acao, arquivo in tqdm(todos_arquivos, desc="Processando áudios"):
    
    nome_acao = acao.name

    pasta_saida = PASTA_SAIDA / nome_acao
    pasta_saida.mkdir(exist_ok=True)

    if nome_acao not in contadores:
        contadores[nome_acao] = 1

    numero = contadores[nome_acao]
    novo_nome = f"{nome_acao}_{numero}.wav"
    output_file = pasta_saida / novo_nome

    processar_audio(arquivo, output_file)

    contadores[nome_acao] += 1


# === RELATÓRIO FINAL ===
print("\n===== RELATÓRIO FINAL =====")
print(f"✅ Processados: {total_processados}")
print(f"⚠️ Ignorados (muito pequenos): {len(arquivos_ignorados)}")
print(f"❌ Erros: {len(arquivos_com_erro)}")

if arquivos_ignorados:
    print("\nArquivos ignorados:")
    for arq in arquivos_ignorados:
        print(arq)

if arquivos_com_erro:
    print("\nArquivos com erro:")
    for arq, erro in arquivos_com_erro:
        print(f"{arq} → {erro}")