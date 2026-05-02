# import os
# import numpy as np
# import soundfile as sf
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from utils import stft_manual, mel_filterbank


# DATASET_PATH = "dataset_final"
# SAMPLE_RATE = 16000
# N_FFT = 1024
# HOP = 512
# N_MELS = 128
# MAX_LEN = 94



# def pad_or_trim(spec, max_len=94):
#     if spec.shape[1] < max_len:
#         pad = max_len - spec.shape[1]
#         spec = np.pad(spec, ((0,0),(0,pad)))
#     else:
#         spec = spec[:, :max_len]
#     return spec


# def gerar_mel(audio, sr):
#     spec = stft_manual(audio, frame_size=N_FFT, hop_size=HOP)

#     spec = spec.T  

#     mel_fb = mel_filterbank(sr, N_FFT, N_MELS)

#     mel = np.dot(mel_fb, spec)

#     mel = np.log(mel + 1e-9)


#     mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-9)

#     mel = pad_or_trim(mel, MAX_LEN)

#     return mel



# X = []
# y = []

# labels = sorted(os.listdir(DATASET_PATH))
# label_map = {label: i for i, label in enumerate(labels)}

# for label in labels:
#     pasta = os.path.join(DATASET_PATH, label)

#     for file in os.listdir(pasta):
#         if not file.endswith(".wav"):
#             continue

#         try:
#             path = os.path.join(pasta, file)

#             audio, sr = sf.read(path)

#             if len(audio.shape) > 1:
#                 audio = np.mean(audio, axis=1)

       
#             if sr != SAMPLE_RATE:
#                 tempo = np.linspace(0, len(audio)/sr, int(len(audio)*SAMPLE_RATE/sr))
#                 audio = np.interp(
#                     tempo,
#                     np.linspace(0, len(audio)/sr, len(audio)),
#                     audio
#                 )
#                 sr = SAMPLE_RATE

#             mel = gerar_mel(audio, sr)

#             mel = np.expand_dims(mel, axis=-1)

#             X.append(mel)
#             y.append(label_map[label])

#         except Exception as e:
#             print(f"Erro: {file} -> {e}")

# X = np.array(X)
# y = np.array(y)



# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# y_train = to_categorical(y_train)
# y_val = to_categorical(y_val)

# print("Treino:", X_train.shape)
# print("Validação:", X_val.shape)



# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# input_shape = (128, 94, 1)
# num_classes = len(labels)

# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
#     MaxPooling2D(2,2),

#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D(2,2),

#     Conv2D(128, (3,3), activation='relu'),
#     MaxPooling2D(2,2),

#     Flatten(),

#     Dense(128, activation='relu'),
#     Dropout(0.3),

#     Dense(num_classes, activation='softmax')
# ])

# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=30,
#     batch_size=32
# )


# from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns
# import matplotlib.pyplot as plt

# preds = model.predict(X_val)
# classes_pred = np.argmax(preds, axis=1)

# y_val_labels = np.argmax(y_val, axis=1)

# print(classification_report(y_val_labels, classes_pred))

# cm = confusion_matrix(y_val_labels, classes_pred)

# plt.figure(figsize=(6,5))
# sns.heatmap(cm, annot=True, fmt='d',
#             xticklabels=labels,
#             yticklabels=labels)
# plt.xlabel("Predito")
# plt.ylabel("Real")
# plt.show()


# model.save("modelo_audio_manual.h5")

# import json
# with open("labels_manual.json", "w") as f:
#     json.dump(label_map, f)

# print("Modelo salvo!")

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = "spectrograms"

# ---------------------------
# 📊 CARREGAR DADOS
# ---------------------------
X = []
y = []

labels = sorted(os.listdir(DATASET_PATH))
label_map = {label: i for i, label in enumerate(labels)}

for label in labels:
    pasta = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(pasta):
        continue

    for file in os.listdir(pasta):
        if file.endswith(".npy"):
            mel = np.load(os.path.join(pasta, file))
            mel = np.expand_dims(mel, axis=-1)

            X.append(mel)
            y.append(label_map[label])

X = np.array(X)
y = np.array(y)

# ---------------------------
# ⚖️ SPLIT (IMPORTANTE)
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------
# 🔢 ONE HOT
# ---------------------------
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

print("Treino:", X_train.shape)
print("Validação:", X_val.shape)

# ---------------------------
# 🧠 MODELO CNN
# ---------------------------
input_shape = X_train.shape[1:]
num_classes = len(labels)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])


model.compile(
    optimizer='adam',
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32
)

# ---------------------------
# 💾 SALVAR
# ---------------------------
model.save("test_beta_model.h5")

import json
with open("test_beta_model_labels.json", "w") as f:
    json.dump(label_map, f)

print("Modelo salvo!")

# ===========================
# 🔮 PREDIÇÃO INTELIGENTE
# ===========================

def entropia(p):
    return -np.sum(p * np.log(p + 1e-9))


def prever(audio_mel, model, label_map, threshold=0.7, entropia_max=1.5):
    """
    audio_mel: espectrograma já processado (128x94)
    """

    labels_inv = {v: k for k, v in label_map.items()}

    x = np.expand_dims(audio_mel, axis=(0, -1))

    pred = model.predict(x, verbose=0)[0]

    prob = np.max(pred)
    classe = np.argmax(pred)
    H = entropia(pred)

    nome_classe = labels_inv[classe]

    # 🔥 REGRA FINAL
    if nome_classe == "ruido":
        return "desconhecido", prob, H

    if prob < threshold or H > entropia_max:
        return "desconhecido", prob, H

    return nome_classe, prob, H

preds = model.predict(X_val)
y_pred = np.argmax(preds, axis=1)
y_true = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=labels,
            yticklabels=labels)

plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()