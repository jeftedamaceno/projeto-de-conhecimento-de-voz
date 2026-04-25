import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

DATASET_PATH = "spectrograms"
IMG_SIZE = (128, 128)

def calcular_media_classe(path_classe):
    imagens = []
    
    for arquivo in os.listdir(path_classe):
        caminho = os.path.join(path_classe, arquivo)
        img = cv2.imread(caminho)
        
        if img is None:
            continue
            
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        imagens.append(img)
    
    return np.mean(imagens, axis=0)

classes = os.listdir(DATASET_PATH)

plt.figure(figsize=(12, 8))

for i, classe in enumerate(classes):
    media = calcular_media_classe(os.path.join(DATASET_PATH, classe))
    
    plt.subplot(2, 3, i+1)
    plt.imshow(media)
    plt.title(f"Média: {classe}")
    plt.axis("off")

plt.show()

def histograma_classe(path_classe):
    pixels = []
    
    for arquivo in os.listdir(path_classe):
        caminho = os.path.join(path_classe, arquivo)
        img = cv2.imread(caminho, 0)  # grayscale
        
        if img is None:
            continue
            
        pixels.extend(img.flatten())
    
    return pixels

plt.figure(figsize=(10, 6))

for classe in classes:
    pixels = histograma_classe(os.path.join(DATASET_PATH, classe))
    plt.hist(pixels, bins=50, alpha=0.5, label=classe)

plt.legend()
plt.title("Distribuição de Intensidade dos Pixels por Classe")
plt.xlabel("Valor do Pixel")
plt.ylabel("Frequência")
plt.show()

def heatmap_classe(path_classe):
    soma = None
    count = 0
    
    for arquivo in os.listdir(path_classe):
        caminho = os.path.join(path_classe, arquivo)
        img = cv2.imread(caminho, 0)
        
        if img is None:
            continue
            
        img = cv2.resize(img, IMG_SIZE)
        
        if soma is None:
            soma = np.zeros_like(img, dtype=float)
        
        soma += img
        count += 1
    
    return soma / count

plt.figure(figsize=(12, 8))

for i, classe in enumerate(classes):
    heatmap = heatmap_classe(os.path.join(DATASET_PATH, classe))
    
    plt.subplot(2, 3, i+1)
    plt.imshow(heatmap, cmap='hot')
    plt.title(f"Heatmap: {classe}")
    plt.axis("off")

plt.show()

from sklearn.decomposition import PCA

X = []
y = []

for idx, classe in enumerate(classes):
    path = os.path.join(DATASET_PATH, classe)
    
    for arquivo in os.listdir(path):
        caminho = os.path.join(path, arquivo)
        img = cv2.imread(caminho, 0)
        
        if img is None:
            continue
            
        img = cv2.resize(img, IMG_SIZE)
        X.append(img.flatten())
        y.append(idx)

X = np.array(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
for i, classe in enumerate(classes):
    idxs = np.array(y) == i
    plt.scatter(X_pca[idxs, 0], X_pca[idxs, 1], label=classe)

plt.legend()
plt.title("PCA das Classes")
plt.show()

for classe in classes:
    pixels = histograma_classe(os.path.join(DATASET_PATH, classe))
    print(classe, "→ média:", np.mean(pixels), "desvio:", np.std(pixels))