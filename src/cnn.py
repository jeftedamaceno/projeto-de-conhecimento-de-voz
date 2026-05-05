import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    Dense, Dropout
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


DATASET_ORIGINAL = "spectrograms_original"
DATASET_RUIDO = "spectrograms_ruido"


X_orig, y_orig = [], []

labels = sorted(os.listdir(DATASET_ORIGINAL))
label_map = {label: i for i, label in enumerate(labels)}

for label in labels:
    path = os.path.join(DATASET_ORIGINAL, label)

    if not os.path.isdir(path):
        continue

    for file in os.listdir(path):
        if file.endswith(".npy"):
            mel = np.load(os.path.join(path, file))
            mel = np.expand_dims(mel, axis=-1)

            X_orig.append(mel)
            y_orig.append(label_map[label])

X_orig = np.array(X_orig)
y_orig = np.array(y_orig)



X_train_orig, X_val, y_train_orig, y_val = train_test_split(
    X_orig, y_orig,
    test_size=0.2,
    stratify=y_orig,
    random_state=42
)


X_aug, y_aug = [], []

for label in labels:
    path = os.path.join(DATASET_RUIDO, label)

    if not os.path.isdir(path):
        continue

    for file in os.listdir(path):
        if file.endswith(".npy"):
            mel = np.load(os.path.join(path, file))
            mel = np.expand_dims(mel, axis=-1)

            X_aug.append(mel)
            y_aug.append(label_map[label])

X_aug = np.array(X_aug)
y_aug = np.array(y_aug)


MAX_AUG_PER_CLASS = 800

X_aug_bal, y_aug_bal = [], []

for label in range(len(labels)):
    idx = np.where(y_aug == label)[0]
    np.random.shuffle(idx)
    idx = idx[:MAX_AUG_PER_CLASS]

    X_aug_bal.append(X_aug[idx])
    y_aug_bal.append(y_aug[idx])

X_aug_bal = np.concatenate(X_aug_bal)
y_aug_bal = np.concatenate(y_aug_bal)


X_train = np.concatenate([X_train_orig, X_aug_bal])
y_train = np.concatenate([y_train_orig, y_aug_bal])


idx = np.random.permutation(len(X_train))
X_train = X_train[idx]
y_train = y_train[idx]

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

print("Treino:", X_train.shape)
print("Val:", X_val.shape)



input_shape = X_train.shape[1:]
num_classes = len(labels)

model = Sequential([

    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), padding='same', activation='relu'),
    BatchNormalization(),

    GlobalAveragePooling2D(),

    Dense(128, activation='relu'),
    Dropout(0.4),

    Dense(num_classes, activation='softmax')
])


model.compile(
    optimizer='adam',
    loss=CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)


callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.3)
]


model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=callbacks
)


model.save("modelo_mic_simulacao1.h5")

import json
with open("labels_mic_simulacao1.json", "w") as f:
    json.dump(label_map, f)

print("Modelo salvo!")



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

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=callbacks
)

# LOSS
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title("Loss")
plt.legend()
plt.show()

# ACCURACY
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title("Accuracy")
plt.legend()
plt.show()

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

y_val_bin = label_binarize(y_true, classes=range(num_classes))

for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_val_bin[:, i], preds[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'Classe {labels[i]} (AUC={roc_auc:.2f})')

plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.show()

from sklearn.metrics import classification_report
import pandas as pd

report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
df = pd.DataFrame(report).transpose()

df[['precision','recall','f1-score']].plot(kind='bar', figsize=(10,5))
plt.title("Precision / Recall / F1")
plt.xticks(rotation=45)
plt.show()

confidences = np.max(preds, axis=1)

plt.hist(confidences, bins=20)
plt.title("Confiança das previsões")
plt.xlabel("Probabilidade")
plt.ylabel("Frequência")
plt.show()

from sklearn.decomposition import PCA

features = model.predict(X_val)

pca = PCA(n_components=2)
reduced = pca.fit_transform(features)

plt.scatter(reduced[:,0], reduced[:,1], c=y_true)
plt.title("Separação das classes (PCA)")
plt.colorbar()
plt.show()