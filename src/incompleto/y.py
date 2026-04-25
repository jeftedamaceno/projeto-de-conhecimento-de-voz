import os
import random
import shutil
from pathlib import Path

# =========================
# CONFIG
# =========================
INPUT_DIR = "spectrograms"
OUTPUT_DIR = "data"

SPLIT = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

random.seed(42)

# =========================
# CRIAR ESTRUTURA
# =========================
for split in SPLIT:
    for classe in os.listdir(INPUT_DIR):
        path = os.path.join(OUTPUT_DIR, split, classe)
        os.makedirs(path, exist_ok=True)

# =========================
# PROCESSAR CADA CLASSE
# =========================
for classe in os.listdir(INPUT_DIR):
    class_path = os.path.join(INPUT_DIR, classe)

    files = os.listdir(class_path)
    random.shuffle(files)

    total = len(files)

    n_train = int(total * SPLIT["train"])
    n_val = int(total * SPLIT["val"])

    train_files = files[:n_train]
    val_files = files[n_train:n_train+n_val]
    test_files = files[n_train+n_val:]

    print(f"{classe}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    # copiar arquivos
    for f in train_files:
        shutil.copy(
            os.path.join(class_path, f),
            os.path.join(OUTPUT_DIR, "train", classe, f)
        )

    for f in val_files:
        shutil.copy(
            os.path.join(class_path, f),
            os.path.join(OUTPUT_DIR, "val", classe, f)
        )

    for f in test_files:
        shutil.copy(
            os.path.join(class_path, f),
            os.path.join(OUTPUT_DIR, "test", classe, f)
        )

print("Dataset organizado com sucesso!")