import numpy as np
import math


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

def time_stretch(signal, factor):
    signal = np.array(signal)
    
    indices = np.arange(0, len(signal), factor)
    return np.interp(indices, np.arange(len(signal)), signal)

def muda_pitch(signal, n_steps):
    signal = np.array(signal)
    
    factor = 2 ** (n_steps / 12)
    
    stretched = time_stretch(signal, 1 / factor)

    result = np.interp(
        np.linspace(0, len(stretched)-1, len(signal)),
        np.arange(len(stretched)),
        stretched
    )
    
    return result

def istft(spectrogram, frame_size=1024, hop=512):
    signal_len = (len(spectrogram) * hop) + frame_size
    signal = np.zeros(signal_len)
    window = np.hanning(frame_size)
    
    for i, frame in enumerate(spectrogram):
        start = i * hop
        signal[start:start+frame_size] += np.real(np.fft.ifft(frame)) * window
    
    return signal

def pad_or_trim(audio, target_len):
    if len(audio) > target_len:
        return audio[:target_len]
    elif len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)))
    return audio

# funcoes para resolver problemas de argumentação forte e vies no treinamento n e certeza que vao ficar ate o finaal do projeto


def normalize_rms(audio, target_rms=0.1):
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        audio = audio * (target_rms / rms)
    return audio



def pre_emphasis(audio, alpha=0.97):
    return np.append(audio[0], audio[1:] - alpha * audio[:-1])



def remove_silence(audio, threshold=0.02):
    energia = np.abs(audio)
    indices = np.where(energia > threshold)[0]

    if len(indices) > 0:
        audio = audio[indices[0]:indices[-1]]

    return audio


def pad_or_trim(audio, target_length):
    if len(audio) < target_length:
        pad = target_length - len(audio)
        audio = np.pad(audio, (0, pad))
    else:
        audio = audio[:target_length]
    return audio


def add_noise(audio, noise_factor=0.001):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise


def random_gain(audio):
    gain = np.random.uniform(0.8, 1.2)
    return audio * gain


def time_stretch(audio, rate=1.0):
    indices = np.arange(0, len(audio), rate)
    indices = indices[indices < len(audio)].astype(int)
    return audio[indices]


def muda_pitch(audio, shift):
    return np.roll(audio, shift)



def preprocess_audio(audio, target_length):

    audio = remove_silence(audio)

    audio = pad_or_trim(audio, target_length)

    audio = pre_emphasis(audio)

    audio = normalize_rms(audio)

    return audio