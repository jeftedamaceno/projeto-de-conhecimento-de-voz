import os
import librosa
import numpy as np
import soundfile as sf


INPUT_DIR = "dataset"
OUTPUT_DIR = "dataset_ruido"

SAMPLE_RATE = 16000


NOISE_FACTOR = 0.003

os.makedirs(OUTPUT_DIR, exist_ok=True)

def add_noise(audio):
    noise = np.random.randn(len(audio))
    audio_noisy = audio + NOISE_FACTOR * noise
    return np.clip(audio_noisy, -1, 1)


def augment_audio(audio, sr):
    audios = []


    audios.append(audio)

    audios.append(add_noise(audio))

    audios.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=2))

    audios.append(librosa.effects.time_stretch(audio, rate=0.9))

    return audios



for label in os.listdir(INPUT_DIR):
    input_label_path = os.path.join(INPUT_DIR, label)

    if not os.path.isdir(input_label_path):
        continue

    output_label_path = os.path.join(OUTPUT_DIR, label)
    os.makedirs(output_label_path, exist_ok=True)

    contador = 0

    for file in os.listdir(input_label_path):
        file_path = os.path.join(input_label_path, file)

        try:
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

           
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

            versões = augment_audio(audio, sr)

            for i, aug_audio in enumerate(versões):
                nome = f"{os.path.splitext(file)[0]}_aug_{i}.wav"
                caminho_saida = os.path.join(output_label_path, nome)

                sf.write(caminho_saida, aug_audio, SAMPLE_RATE)

                contador += 1

        except Exception as e:
            print(f"Erro em {file_path}: {e}")

    print(f"{label}: {contador} arquivos gerados")

print("Dataset com ruído criado!")