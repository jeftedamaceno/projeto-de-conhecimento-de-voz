import numpy as np


def normalizar(audio):
    max_val = np.max(np.abs(audio))
    return audio / max_val if max_val > 0 else audio



def add_noise(audio, noise_factor=0.003):
    noise = np.random.randn(len(audio))
    return np.clip(audio + noise_factor * noise, -1, 1)


def time_stretch(audio, rate=1.0):
    indices = np.arange(0, len(audio), rate)
    indices = indices[indices < len(audio)].astype(int)
    return audio[indices]



def pitch_shift(audio, n_steps):
    fator = 2 ** (n_steps / 12)
    stretched = time_stretch(audio, 1 / fator)

    indices = np.linspace(0, len(stretched)-1, len(audio)).astype(int)
    return stretched[indices]



def stft(audio, frame_size=1024, hop_size=512):
    frames = []

    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i+frame_size]

      
        window = np.hamming(frame_size)
        frame = frame * window

        fft = np.fft.fft(frame)
        frames.append(np.abs(fft[:frame_size // 2]))

    return np.array(frames)



def filtro_voz_stft(audio, sr):
    espectrograma = stft(audio)

    freqs = np.fft.fftfreq(1024, 1/sr)[:512]

  
    mask = (freqs >= 300) & (freqs <= 4000)

    espectrograma_filtrado = espectrograma * mask

    audio_filtrado = np.zeros(len(audio))

    frame_size = 1024
    hop_size = 512

    for i, frame in enumerate(espectrograma_filtrado):
        full_fft = np.concatenate([frame, frame[::-1]])
        rec = np.real(np.fft.ifft(full_fft))

        start = i * hop_size
        audio_filtrado[start:start+frame_size] += rec[:frame_size]

    return normalizar(audio_filtrado)