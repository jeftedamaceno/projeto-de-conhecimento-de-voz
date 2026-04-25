import tensorflow as tf
from tensorflow.keras import layers, models
IMG_SIZE = (128, 128)
BATCH_SIZE = 16

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "spectrograms",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "spectrograms",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42
)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    data_augmentation,

    layers.Rescaling(1./255),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(5, activation='softmax')  # 5 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

loss, acc = model.evaluate(val_ds)
print("Accuracy:", acc)

# import tensorflow as tf
# from tensorflow.keras import layers, models
# import os

# IMG_SIZE = (128, 128)
# BATCH_SIZE = 16

# # ==============================
# # DATASET (treino + validação)
# # ==============================

# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "spectrograms",
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     validation_split=0.2,
#     subset="training",
#     seed=42
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "spectrograms",
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     validation_split=0.2,
#     subset="validation",
#     seed=42
# )

# class_names = train_ds.class_names
# print("Classes:", class_names)

# # ==============================
# # PERFORMANCE (otimização)
# # ==============================

# AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.cache().shuffle(500).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# # ==============================
# # DATA AUGMENTATION (CORRIGIDO)
# # ==============================

# # data_augmentation = tf.keras.Sequential([
# #     layers.RandomZoom(0.1),
# #     layers.RandomContrast(0.1),
# # ])
# data_augmentation = tf.keras.Sequential([])

# # ==============================
# # MODELO CNN MELHORADO
# # ==============================

# model = models.Sequential([
#     layers.Input(shape=(128, 128, 3)),

#     # data_augmentation,
#     layers.Rescaling(1./255),

#     # BLOCO 1
#     layers.Conv2D(32, (3,3), padding='same'),
#     layers.BatchNormalization(),
#     layers.Activation('relu'),
#     layers.MaxPooling2D(),
#     layers.Dropout(0.2),

#     # BLOCO 2
#     layers.Conv2D(64, (3,3), padding='same'),
#     layers.BatchNormalization(),
#     layers.Activation('relu'),
#     layers.MaxPooling2D(),
#     layers.Dropout(0.3),

#     # BLOCO 3
#     layers.Conv2D(128, (3,3), padding='same'),
#     layers.BatchNormalization(),
#     layers.Activation('relu'),
#     layers.MaxPooling2D(),
#     layers.Dropout(0.3),

#     layers.Flatten(),

#     layers.Dense(128, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dropout(0.4),

#     layers.Dense(len(class_names), activation='softmax')
# ])

# # ==============================
# # COMPILAÇÃO
# # ==============================

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.summary()

# # ==============================
# # CALLBACKS IMPORTANTES
# # ==============================

# callbacks = [
#     tf.keras.callbacks.EarlyStopping(
#         monitor='val_loss',
#         patience=4,
#         restore_best_weights=True
#     ),

#     tf.keras.callbacks.ModelCheckpoint(
#         "modelo_voz.h5",
#         monitor="val_accuracy",
#         save_best_only=True
#     )
# ]

# # ==============================
# # TREINO
# # ==============================

# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=30,
#     callbacks=callbacks
# )

# # ==============================
# # SALVAR CLASSES
# # ==============================

# import json
# with open("classes.json", "w") as f:
#     json.dump(class_names, f)

# # for images, labels in val_ds.unbatch().take(20):
# #     print(labels.numpy())
# import numpy as np

# train_counts = {}
# val_counts = {}

# for _, labels in train_ds.unbatch():
#     l = int(labels.numpy())
#     train_counts[l] = train_counts.get(l, 0) + 1

# for _, labels in val_ds.unbatch():
#     l = int(labels.numpy())
#     val_counts[l] = val_counts.get(l, 0) + 1

# print("Treino:", train_counts)
# print("Validação:", val_counts)

# import matplotlib.pyplot as plt

# print("TREINO:")
# for images, labels in train_ds.take(1):
#     for i in range(3):
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(f"Classe: {labels[i].numpy()}")
#         plt.show()

# print("VALIDAÇÃO:")
# for images, labels in val_ds.take(1):
#     for i in range(3):
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(f"Classe: {labels[i].numpy()}")
#         plt.show()

# loss, acc = model.evaluate(train_ds)
# print("Train accuracy:", acc)

# print("Modelo salvo com sucesso!")