import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from PIL import Image

# Obtén la ruta del directorio del archivo actual
dir_path = os.path.dirname(os.path.realpath(__file__))

# Construye la ruta al directorio de imágenes
image_dir = os.path.join(dir_path, 'datasets_imgs_01')

# Listar los archivos de imágenes en el directorio
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]

# Cargar las imágenes y convertirlas en vectores
image_vectors = []
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path).convert('L')  # Convertir a escala de grises
    image_array = np.array(image)
    image_vector = image_array.flatten()
    image_vectors.append(image_vector)

# Verificar si se cargaron imágenes
if not image_vectors:
    raise ValueError("No images were loaded.")

# Convertir la lista de vectores en una matriz de datos
X = np.array(image_vectors)

# Verificar si las imágenes son de un solo píxel
if X.ndim == 1:
    X = X.reshape(-1, 1)

# Centrar los datos
X_centered = X - np.mean(X, axis=0)

# Aplicar SVD para reducir la dimensionalidad
n_components = 50  # Número de componentes principales deseados
svd = TruncatedSVD(n_components=n_components)
X_reduced = svd.fit_transform(X_centered)

# Visualizar la variación explicada por los componentes principales
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(svd.explained_variance_ratio_), marker='o')
plt.xlabel('Número de componentes')
plt.ylabel('Varianza explicada acumulada')
plt.title('Variación explicada por los componentes principales')
plt.grid()
plt.show()

# Imprimir las primeras imágenes reconstruidas a partir de la representación reducida
n_images_to_show = 5
fig, axes = plt.subplots(2, n_images_to_show, figsize=(15, 6))

for i in range(n_images_to_show):
    # Imagen original
    axes[0, i].imshow(X[i].reshape(image_array.shape), cmap='gray')
    axes[0, i].set_title('Original')
    axes[0, i].axis('off')
    
    # Reconstruir la imagen a partir de los componentes principales
    reconstructed_image = svd.inverse_transform(X_reduced[i].reshape(1, -1)).reshape(image_array.shape)
    axes[1, i].imshow(reconstructed_image, cmap='gray')
    axes[1, i].set_title('Reconstruida')
    axes[1, i].axis('off')

plt.suptitle('Imágenes originales y reconstruidas')
plt.show()
