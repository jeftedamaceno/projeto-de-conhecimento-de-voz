import numpy as np

def standardize_audio(audio):
    audio = audio.astype(np.float32)

    if np.max(np.abs(audio)) > 1:
        audio = audio / 32768.0

    return audio


def normalize_rms(audio, target_rms=0.1):
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        audio = audio * (target_rms / rms)
    return audio


def pre_emphasis(audio, alpha=0.97):
    return np.append(audio[0], audio[1:] - alpha * audio[:-1])


def remove_silence(audio, threshold=0.01):
    energia = np.abs(audio)
    indices = np.where(energia > threshold)[0]

    if len(indices) > 0:
        return audio[indices[0]:indices[-1]]

    return audio


def simulate_microphone(audio):


    noise = np.random.randn(len(audio))
    audio = audio + 0.003 * noise


    gain = np.random.uniform(0.5, 2.0)
    audio = audio * gain

   
    audio = np.clip(audio, -1.0, 1.0)

    return audio



def pad_or_trim(audio, target_length):
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    return audio


def preprocess_audio(audio, target_length, training=False):

    audio = standardize_audio(audio)

    audio = remove_silence(audio)

    audio = pad_or_trim(audio, target_length)

    audio = pre_emphasis(audio)

    audio = normalize_rms(audio)

    if training:
        audio = simulate_microphone(audio)

    return audio



def stft_manual(signal, n_fft=512, hop_length=160):
    frames = []

    for i in range(0, len(signal) - n_fft, hop_length):
        frame = signal[i:i+n_fft]
        window = np.hanning(n_fft)
        frame = frame * window
        spectrum = np.fft.rfft(frame)
        frames.append(spectrum)

    return np.array(frames)



def mel_filterbank(sr, n_fft, n_mels=128):

    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(sr/2), n_mels+2)
    hz_points = mel_to_hz(mel_points)

    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, int(n_fft/2 + 1)))

    for i in range(1, n_mels + 1):
        for j in range(bins[i-1], bins[i]):
            fb[i-1, j] = (j - bins[i-1]) / (bins[i] - bins[i-1])
        for j in range(bins[i], bins[i+1]):
            fb[i-1, j] = (bins[i+1] - j) / (bins[i+1] - bins[i])

    return fb

import numpy as np

def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_level * noise

def random_shift(audio, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(audio))
    return np.roll(audio, shift)

def random_gain(audio):
    gain = np.random.uniform(0.7, 1.3)
    return audio * gain