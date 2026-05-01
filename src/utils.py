# import numpy as np


# def normalizar(audio):
#     max_val = np.max(np.abs(audio))
#     return audio / max_val if max_val > 0 else audio



# def add_noise(audio, noise_factor=0.003):
#     noise = np.random.randn(len(audio))
#     return np.clip(audio + noise_factor * noise, -1, 1)


# def time_stretch(audio, rate=1.0):
#     indices = np.arange(0, len(audio), rate)
#     indices = indices[indices < len(audio)].astype(int)
#     return audio[indices]



# def pitch_shift(audio, n_steps):
#     fator = 2 ** (n_steps / 12)
#     stretched = time_stretch(audio, 1 / fator)

#     indices = np.linspace(0, len(stretched)-1, len(audio)).astype(int)
#     return stretched[indices]



# def stft_manual(audio, frame_size=1024, hop_size=512):
#     frames = []

#     for i in range(0, len(audio) - frame_size, hop_size):
#         frame = audio[i:i+frame_size]

      
#         window = np.hamming(frame_size)
#         frame = frame * window

#         fft = np.fft.fft(frame)
#         frames.append(np.abs(fft[:frame_size // 2]))

#     return np.array(frames)



# def filtro_voz_stft(audio, sr):
#     espectrograma = stft(audio)

#     freqs = np.fft.fftfreq(1024, 1/sr)[:512]

  
#     mask = (freqs >= 300) & (freqs <= 4000)

#     espectrograma_filtrado = espectrograma * mask

#     audio_filtrado = np.zeros(len(audio))

#     frame_size = 1024
#     hop_size = 512

#     for i, frame in enumerate(espectrograma_filtrado):
#         full_fft = np.concatenate([frame, frame[::-1]])
#         rec = np.real(np.fft.ifft(full_fft))

#         start = i * hop_size
#         audio_filtrado[start:start+frame_size] += rec[:frame_size]

#     return normalizar(audio_filtrado)

# def hz_to_mel(hz):
#     return 2595 * np.log10(1 + hz / 700)


# def mel_to_hz(mel):
#     return 700 * (10**(mel / 2595) - 1)


# def mel_filterbank(sr, n_fft, n_mels=128, fmin=0, fmax=None):
#     if fmax is None:
#         fmax = sr / 2

#     mel_min = hz_to_mel(fmin)
#     mel_max = hz_to_mel(fmax)

#     mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
#     hz_points = mel_to_hz(mel_points)

#     bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

#     fb = np.zeros((n_mels, n_fft // 2))

#     for i in range(1, n_mels + 1):
#         left = bins[i - 1]
#         center = bins[i]
#         right = bins[i + 1]

#         for j in range(left, center):
#             fb[i - 1, j] = (j - left) / (center - left + 1e-10)

#         for j in range(center, right):
#             fb[i - 1, j] = (right - j) / (right - center + 1e-10)

#     return fb
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



def stft_manual(audio, n_fft=1024, hop_length=512):
    frames = []

    for i in range(0, len(audio) - n_fft, hop_length):
        frame = audio[i:i+n_fft]

        window = np.hamming(n_fft)
        frame = frame * window

        fft = np.fft.fft(frame)
        frames.append(np.abs(fft[:n_fft // 2]))

    return np.array(frames)


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)


def mel_filterbank(sr, n_fft, n_mels=128, fmin=0, fmax=None):
    if fmax is None:
        fmax = sr / 2

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)

    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2))

    for i in range(1, n_mels + 1):
        left = bins[i - 1]
        center = bins[i]
        right = bins[i + 1]

        for j in range(left, center):
            fb[i - 1, j] = (j - left) / (center - left + 1e-10)

        for j in range(center, right):
            fb[i - 1, j] = (right - j) / (right - center + 1e-10)

    return fb



def gerar_log_mel_spectrogram(audio, sr, n_fft=1024, hop_length=512, n_mels=128):
    S = stft_manual(audio, n_fft=n_fft, hop_length=hop_length)

    mel_fb = mel_filterbank(sr, n_fft=n_fft, n_mels=n_mels)

    mel_spec = np.dot(S, mel_fb.T)

    log_mel = np.log(mel_spec + 1e-9)

    return log_mel