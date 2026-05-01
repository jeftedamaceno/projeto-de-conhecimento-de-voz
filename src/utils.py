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
import math
import wave


def normalizar(audio):
    max_val = np.max(np.abs(audio))
    return audio / max_val if max_val > 0 else audio


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



def get_duracao(audio, sr):
    return len(audio) / sr

def cruzamento_zero(audio):
    cruzamentos = 0
    for i in range(1, len(audio)):
        if audio[i-1] * audio[i] < 0:
            cruzamentos += 1
    return cruzamentos / len(audio)

def rms(signal):
    return math.sqrt(sum(x*x for x in signal) / len(signal))


def istft(spectrogram, frame_size=1024, hop=512):
    signal_len = (len(spectrogram) * hop) + frame_size
    signal = np.zeros(signal_len)
    window = np.hanning(frame_size)
    
    for i, frame in enumerate(spectrogram):
        start = i * hop
        signal[start:start+frame_size] += np.real(np.fft.ifft(frame)) * window
    
    return signal


def phase_vocoder(spec, stretch, hop=512):
    n_frames, n_bins = spec.shape
    
    # saída
    time_steps = np.arange(0, n_frames, stretch)
    
    output = np.zeros((len(time_steps), n_bins), dtype=np.complex64)
    
    # fase acumulada
    phase_acc = np.angle(spec[0])
    
    # fase anterior
    prev_phase = np.angle(spec[0])
    
    for i, t in enumerate(time_steps):
        t0 = int(np.floor(t))
        t1 = min(t0 + 1, n_frames - 1)
        
        #interpolação de magnitude
        mag = (1 - (t - t0)) * np.abs(spec[t0]) + (t - t0) * np.abs(spec[t1])
        
        #fase atual
        phase = np.angle(spec[t1])
        
        #diferença das fases
        delta = phase - prev_phase
        
        #remover avanço esperado
        k = np.arange(n_bins)
        expected = 2 * np.pi * k * hop / n_bins
        
        delta -= expected
        
        #wrap para [-pi, pi]
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        
        # adicionar avanço esperado de volta
        phase_acc += expected + delta
        
        output[i] = mag * np.exp(1j * phase_acc)
        
        prev_phase = phase

    return output


def time_stretch(audio, stretch, frame_size=1024, pulo=512):
    spec = stft(audio, frame_size, pulo)
    stretched_spec = phase_vocoder(spec, stretch, pulo)
    return istft(stretched_spec, frame_size, pulo)


def pitch_shift(signal, n_steps, sr):
    factor = 2 ** (n_steps / 12)
    
    # 1. muda duração sem alterar pitch
    stretched = time_stretch(signal, 1 / factor)
    
    # 2. reamostra para tamanho original
    result = np.interp(
        np.linspace(0, len(stretched)-1, len(signal)),
        np.arange(len(stretched)),
        stretched
    )
    
    return result


def ruido_branco(audio, ruido=0.01):
    audio = np.array(audio)
    
    ruido = np.random.normal(0, ruido, size=audio.shape)
    return audio + ruido


def carrega_wav(path, mono=True):
    with wave.open(path, 'rb') as w:
        sr = w.getframerate()
        n_frames = w.getnframes()
        n_channels = w.getnchannels()
        amostra = w.getsampwidth()
        
        frames = w.readframes(n_frames)
    
    # converte para numpy
    if amostra == 2:
        dtype = np.int16
    elif amostra == 4:
        dtype = np.int32
    else:
        raise ValueError("Formato não suportado")
    
    audio = np.frombuffer(frames, dtype=dtype)
    
    # reshape se for estéreo
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)
        if mono:
            audio = np.mean(audio, axis=1)
    
    # normalização [-1,1]
    audio = audio.astype(np.float32) / np.max(np.abs(audio))
    
    return audio, sr


def resample(audio, orig_sr, target_sr):
    
    if orig_sr == target_sr:
        return audio
    
    duration = len(audio) / orig_sr
    new_length = int(duration * target_sr)
    
    old_indices = np.arange(len(audio))
    new_indices = np.linspace(0, len(audio) - 1, new_length)
    
    return np.interp(new_indices, old_indices, audio)


def carrega_audio(path, sr=None, mono=True):
    audio, orig_sr = carrega_wav(path, mono=mono)
    
    if sr is not None:
        audio = resample(audio, orig_sr, sr)
        return audio, sr
    
    return audio, orig_sr