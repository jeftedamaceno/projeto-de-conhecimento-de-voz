import os
import numpy as np
import matplotlib.pyplot as plt

INPUT_DIR = "dados"
OUTPUT_DIR = "data_img"

for split in ["train", "val", "test"]:
    for classe in os.listdir(os.path.join(INPUT_DIR, split)):
        in_path = os.path.join(INPUT_DIR, split, classe)
        out_path = os.path.join(OUTPUT_DIR, split, classe)

        os.makedirs(out_path, exist_ok=True)

        for file in os.listdir(in_path):
            if file.endswith(".npy"):
                arr = np.load(os.path.join(in_path, file))

                # normalizar para imagem
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

                plt.imsave(
                    os.path.join(out_path, file.replace(".npy", ".png")),
                    arr
                )

print("Conversão concluída!")
