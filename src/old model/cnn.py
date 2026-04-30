import os
import numpy as np

DATASET_PATH = "spectrograms"

X = []
y = []

labels = sorted(os.listdir(DATASET_PATH))
label_map = {label: i for i, label in enumerate(labels)}

print("Labels:", label_map)

for label in labels:
    pasta = os.path.join(DATASET_PATH, label)

    for file in os.listdir(pasta):
        if not file.endswith(".npy"):
            continue

        caminho = os.path.join(pasta, file)
        mel = np.load(caminho)

      
        if mel.shape != (128, 94):
            continue

     
        mel = np.expand_dims(mel, axis=-1)

        X.append(mel)
        y.append(label_map[label])

X = np.array(X)
y = np.array(y)

print("Shape X:", X.shape)

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

input_shape = (128, 94, 1)
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
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32
)
preds = model.predict(X_val)
classes_pred = np.argmax(preds, axis=1)

from collections import Counter
print("Distribuição das previsões:")
print(Counter(classes_pred))
from sklearn.metrics import classification_report

y_val_labels = np.argmax(y_val, axis=1)

print(classification_report(y_val_labels, classes_pred, target_names=labels))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_val_labels, classes_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

model.save("modelo_audio.h5")

import json
with open("labels.json", "w") as f:
    json.dump(label_map, f)

print("Modelo salvo!")