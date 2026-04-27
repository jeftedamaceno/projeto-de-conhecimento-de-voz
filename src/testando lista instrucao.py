import numpy as np
import librosa
import sounddevice as sd
from tensorflow.keras.models import load_model
import json
import time
import keyboard  

# ==============================
# CONFIG
# ==============================
SAMPLE_RATE = 16000
CHUNK_DURATION = 1.2
STEP = 0.6

N_MELS = 128
MAX_LEN = 94

THRESHOLD = 0.75
MIN_REPETICOES = 2  # quantas vezes precisa repetir para confirmar

MODEL_PATH = "modelo_audio_da.h5"
LABELS_PATH = "labels_da.json"

# ==============================
# CARREGAR
# ==============================
model = load_model(MODEL_PATH)

with open(LABELS_PATH) as f:
    label_map = json.load(f)

inv_map = {v: k for k, v in label_map.items()}

# ==============================
# BUFFER
# ==============================
buffer = np.zeros(int(SAMPLE_RATE * CHUNK_DURATION))

# histórico de previsões
historico = []
sequencia_final = []

# ==============================
# MEL
# ==============================
def audio_para_mel(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    if mel_db.shape[1] < MAX_LEN:
        pad = MAX_LEN - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad)))
    else:
        mel_db = mel_db[:, :MAX_LEN]

    return mel_db

# ==============================
# CALLBACK
# ==============================
def callback(indata, frames, time_info, status):
    global buffer, historico, sequencia_final

    audio = indata[:, 0]

    # atualiza buffer
    buffer[:] = np.roll(buffer, -len(audio))
    buffer[-len(audio):] = audio

    mel = audio_para_mel(buffer)

    mel = np.expand_dims(mel, axis=-1)
    mel = np.expand_dims(mel, axis=0)

    pred = model.predict(mel, verbose=0)

    classe = inv_map[np.argmax(pred)]
    confianca = np.max(pred)

    if confianca > THRESHOLD:
        historico.append(classe)

        # manter só últimos valores
        if len(historico) > 5:
            historico.pop(0)

        # verificar repetição
        if historico.count(classe) >= MIN_REPETICOES:
            # evitar duplicado na sequência final
            if len(sequencia_final) == 0 or sequencia_final[-1] != classe:
                sequencia_final.append(classe)
                print(f"✅ Detectado: {classe} ({confianca:.2f})")

# ==============================
# EXECUÇÃO
# ==============================
print("🎤 Ouvindo... pressione 'q' para parar\n")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    blocksize=int(SAMPLE_RATE * STEP),
    callback=callback
):
    while True:
        if keyboard.is_pressed('q'):
            print("\n🛑 Parando...")
            break
        time.sleep(0.1)

# ==============================
# RESULTADO FINAL
# ==============================
print("\n📜 Sequência final detectada:")
print(" -> ".join(sequencia_final))