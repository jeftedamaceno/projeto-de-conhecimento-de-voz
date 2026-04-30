import numpy as np


def normalizar_audio(y):
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    return y


def janela_hamming(N):
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))


def stft(y, n_fft=1024, hop_length=512):
    window = janela_hamming(n_fft)
    frames = []

    for i in range(0, len(y) - n_fft, hop_length):
        frame = y[i:i+n_fft]
        frame = frame * window  # aplica janela

        fft = np.fft.fft(frame)
        frames.append(fft)

    return np.array(frames)  


def espectrograma(y, n_fft=1024, hop_length=512):
    stft_matrix = stft(y, n_fft, hop_length)
    magnitude = np.abs(stft_matrix)
    return magnitude


def log_espectrograma(y, n_fft=1024, hop_length=512):
    spec = espectrograma(y, n_fft, hop_length)
    return np.log(spec + 1e-9)


def filtro_passa_banda(y, sr, f_min=300, f_max=4000):
    fft = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(y), 1/sr)

    fft[(freqs < f_min) | (freqs > f_max)] = 0

    return np.real(np.fft.ifft(fft))


def detectar_energia(y, frame_size=1024, hop=512, threshold=0.02):
    energias = []

    for i in range(0, len(y) - frame_size, hop):
        frame = y[i:i+frame_size]
        energia = np.sum(frame**2)
        energias.append(energia)

    energias = np.array(energias)
    energias = energias / (np.max(energias) + 1e-9)

    indices = np.where(energias > threshold)[0]

    if len(indices) == 0:
        return None

    inicio = indices[0] * hop
    fim = indices[-1] * hop + frame_size

    return inicio, fim