import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read


SAMPLE_RATE = 16000
TARGET_LENGTH = int(1.2 * SAMPLE_RATE)


from utils import (
    stft_manual,
    mel_filterbank,
    add_noise,
    time_stretch,
    muda_pitch,
    pad_or_trim
)


import os
from scipy.io.wavfile import read

def carregar_dataset(path_dataset):

    dados = {}

    for classe in os.listdir(path_dataset):

        caminho_classe = os.path.join(path_dataset, classe)

        if not os.path.isdir(caminho_classe):
            continue

        audios = []

        for arquivo in os.listdir(caminho_classe):
            if arquivo.endswith(".wav"):

                caminho_audio = os.path.join(caminho_classe, arquivo)

                sr, audio = read(caminho_audio)
                audio = audio.astype(np.float32)

                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)

                audios.append(audio)

        dados[classe] = audios

    return dados


def preprocess_audio(audio):

    energia = np.abs(audio)
    indices = np.where(energia > 0.01)[0]

    if len(indices) > 0:
        audio = audio[indices[0]:indices[-1]]

  
    audio = pad_or_trim(audio, TARGET_LENGTH)

    return audio


def audio_para_mel(audio):

    audio = preprocess_audio(audio)

    spec = stft_manual(audio, n_fft=512, hop_length=160)
    spec_power = np.abs(spec) ** 2

    mel_fb = mel_filterbank(sr=SAMPLE_RATE, n_fft=512, n_mels=128)

    mel_spec = np.dot(spec_power, mel_fb.T)

    mel_db = np.log(mel_spec + 1e-6)


    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    return mel_db.T 



def plot_mel(mel, titulo="Spectrograma"):

    plt.figure(figsize=(8, 4))
    plt.imshow(mel, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar()
    plt.title(titulo)
    plt.xlabel("Tempo")
    plt.ylabel("Mel")
    plt.tight_layout()
    plt.show()



def analisar_audio(audio):

   
    mel_original = audio_para_mel(audio)

  
    audio_noise = add_noise(audio, noise_factor=0.003)
    mel_noise = audio_para_mel(audio_noise)

    audio_pitch = muda_pitch(audio, 2)
    mel_pitch = audio_para_mel(audio_pitch)

    audio_stretch = time_stretch(audio, 1.2)
    mel_stretch = audio_para_mel(audio_stretch)

    
    plot_mel(mel_original, "Original")
    plot_mel(mel_noise, "Ruído")
    plot_mel(mel_pitch, "Pitch Shift (+2)")
    plot_mel(mel_stretch, "Time Stretch (1.2x)")


def testar_varios_augmentations(audio, n=6):

    for i in range(n):

        aug = audio.copy()

      
        if np.random.rand() > 0.5:
            aug = add_noise(aug, 0.003)

        if np.random.rand() > 0.5:
            aug = muda_pitch(aug, np.random.randint(-2, 3))

        if np.random.rand() > 0.5:
            aug = time_stretch(aug, np.random.uniform(0.9, 1.1))

        mel = audio_para_mel(aug)

        plot_mel(mel, f"Augmentação {i+1}")


def diferenca(m1, m2):
    return np.mean(np.abs(m1 - m2))



def main():

    dataset_final = carregar_dataset("dataset_final")
    dataset_ruido = carregar_dataset("dataset_ruido_2")

    print("\n=== DATASET LIMPO ===")
    analisar_classes(dataset_final)

    print("\n=== DATASET RUIDO ===")
    analisar_classes(dataset_ruido)

    print("\n=== MÉDIAS MEL ===")
    media_mel_por_classe(dataset_final)

    print("\n=== COMPARAÇÃO ===")
    comparar_datasets(dataset_final, dataset_ruido)

def analisar_classes(dataset):

    for classe, audios in dataset.items():

        duracoes = []
        energias = []

        for audio in audios:
            duracoes.append(len(audio) / SAMPLE_RATE)
            energias.append(np.mean(np.abs(audio)))

        print(f"\nClasse: {classe}")
        print(f"Qtd amostras: {len(audios)}")
        print(f"Duração média: {np.mean(duracoes):.2f}s")
        print(f"Energia média: {np.mean(energias):.4f}")

def media_mel_por_classe(dataset):

    medias = {}

    for classe, audios in dataset.items():

        mels = []

        for audio in audios:
            mel = audio_para_mel(audio)
            mels.append(mel)

        media = np.mean(mels, axis=0)
        medias[classe] = media

        plot_mel(media, f"Média MEL - {classe}")

    return medias

def comparar_datasets(ds1, ds2):

    for classe in ds1.keys():

        m1 = media_mel_por_classe({classe: ds1[classe]})[classe]
        m2 = media_mel_por_classe({classe: ds2[classe]})[classe]

        diff = diferenca(m1, m2)

        print(f"Diferença {classe}: {diff:.4f}")


if __name__ == "__main__":
    main()