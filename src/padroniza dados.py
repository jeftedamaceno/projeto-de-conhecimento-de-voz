from pydub import AudioSegment
from pathlib import Path

def converter_para_wav_padrao(input_file, output_file):
    audio = AudioSegment.from_file(input_file)

    # Padronizando
    audio = audio.set_frame_rate(44100)  # 44.1 kHz
    audio = audio.set_channels(2)        # estéreo

    # Exportando WAV (PCM padrão)
    audio.export(
        output_file,
        format="wav",
        parameters=[
            "-acodec", "pcm_s16le"  # PCM 16 bits (padrão comum)
        ]
    )
    
pasta_entrada = Path('./audios_entrada')
pasta_saida = Path('./audios_saida')

pasta_saida.mkdir(exist_ok=True)

for arquivo in pasta_entrada.iterdir():
    if arquivo.is_file():
        nome_saida = pasta_saida / (arquivo.stem + ".wav")
        converter_para_wav_padrao(arquivo, nome_saida)