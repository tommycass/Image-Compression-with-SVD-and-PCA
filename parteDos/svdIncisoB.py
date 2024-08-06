import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image
import seaborn as sns

# Función para cargar imágenes desde un directorio
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(folder, filename))
            images.append(np.array(img).flatten())
    return np.array(images)

# Función SVD personalizada
def svd(X, num_components):
    U, s, VT = np.linalg.svd(X, full_matrices=False)
    return U[:, :num_components], s[:num_components], VT[:num_components, :]

# Función K-means personalizada
def kmeans(X, n_clusters, random_state=0):
    np.random.seed(random_state)
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
    while True:
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels


def reconstruct_images(U, S, VT):
    return U @ np.diag(S) @ VT

# Función para reconstruir y visualizar imágenes
def plot_reconstructed_images(X, dims, n_images_to_show=9):
    labels = kmeans(X, n_clusters=n_images_to_show)
    distinct_indices = [np.where(labels == i)[0][0] for i in range(n_images_to_show)]
    p = int(np.sqrt(X.shape[1]))

    for d in dims:
        U, s, VT = svd(X, d)
        X_reconstructed = reconstruct_images(U, s, VT)
        fig, axes = plt.subplots(2, n_images_to_show, figsize=(20, 10))
        for i, index in enumerate(distinct_indices):
            axes[0, i].imshow(X[index].reshape((p, p)), cmap='gray')
            axes[0, i].set_title('Original', fontsize=16.5)
            axes[0, i].axis('off')
            axes[1, i].imshow(X_reconstructed[index].reshape((p, p)), cmap='gray')
            axes[1, i].set_title('Reconstruida', fontsize=16.5)
            axes[1, i].axis('off')
        plt.suptitle(r'Imágenes originales y reconstruidas ($d={}$)'.format(d), fontsize=25)
        plt.show()

def plot_reconstructed_images_last_aves(X, dims, n_images_to_show=9):
    labels = kmeans(X, n_clusters=n_images_to_show)
    distinct_indices = [np.where(labels == i)[0][0] for i in range(n_images_to_show)]
    p = int(np.sqrt(X.shape[1]))

    U, s, VT = np.linalg.svd(X, full_matrices=False)
    
    fig, axes = plt.subplots(len(dims) + 1, n_images_to_show, figsize=(20, 5 * (len(dims) + 1)))
    for i, index in enumerate(distinct_indices):
        # Imagen original
        axes[0, i].imshow(X[index].reshape((p, p)), cmap='gray')
        axes[0, i].set_title('Original', fontsize=16)
        axes[0, i].axis('off')

    for row, d in enumerate(dims, start=1):
        # Reconstrucción con los últimos d autovalores menos significativos
        U_d_least = U[:, -d:]
        s_d_least = s[-d:]
        VT_d_least = VT[-d:, :]
        X_reconstructed_last = reconstruct_images(U_d_least, s_d_least, VT_d_least)
        
        for i, index in enumerate(distinct_indices):
            # Imagen reconstruida con los últimos d autovalores menos significativos
            axes[row, i].imshow(X_reconstructed_last[index].reshape((p, p)), cmap='gray')
            axes[row, i].set_title(f'últimos {d}', fontsize=16)
            axes[row, i].axis('off')
        
    plt.suptitle('Imágenes originales y reconstruidas con los últimos autovectores', fontsize=25)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

# Función para comparar imágenes reconstruidas
def plot_reconstructed_images_comparison(X, dims, n_images_to_show=8):
    labels = kmeans(X, n_clusters=n_images_to_show)
    distinct_indices = [np.where(labels == i)[0][0] for i in range(n_images_to_show)]
    p = int(np.sqrt(X.shape[1]))
    fig, axes = plt.subplots(len(dims), n_images_to_show, figsize=(20, 10))

    for j, d in enumerate(dims):
        U, s, VT = svd(X, d)
        X_reconstructed = U @ np.diag(s) @ VT
        for i, index in enumerate(distinct_indices):
            axes[j, i].imshow(X[index].reshape((p, p)), cmap='gray')
            axes[j, i].set_title(r'Reconstruida $d={}$'.format(d), fontsize=12)
            axes[j, i].axis('off')
    plt.suptitle('Comparación de imágenes reconstruidas', fontsize=25)
    plt.show()

# Función para calcular y visualizar la matriz de similaridad
def plot_similarity_matrices(X, dims, sigma):
    for d in dims:
        U, s, VT = svd(X, d)
        X_reduced = U @ np.diag(s) @ VT
        dist_matrix = euclidean_distances(X_reduced)
        similarity_matrix = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(similarity_matrix, cmap='viridis', xticklabels=False, yticklabels=False, cbar_kws={'label': 'Similaridad'})
        plt.title(r'Matriz de Similaridad en Espacio Reducido ($d={}$)'.format(d), fontsize=22)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('Similaridad', size=20)
        plt.show()

def plot_similarity_matrices_sigma(X, dims, sigmas):
    for d in dims:
        U, s, VT = svd(X, d)
        X_reduced = U @ np.diag(s) @ VT
        dist_matrix = euclidean_distances(X_reduced)

        fig, axes = plt.subplots(1, len(sigmas), figsize=(10 * len(sigmas), 8))
        for i, sigma in enumerate(sigmas):
            similarity_matrix = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
            ax = sns.heatmap(similarity_matrix, cmap='viridis', xticklabels=False, yticklabels=False, cbar_kws={'label': 'Similaridad'}, ax=axes[i])
            axes[i].set_title(r'$\sigma={}$'.format(sigma), fontsize=22)
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=20)
            cbar.set_label('Similaridad', size=20)
        fig.suptitle(r'Matrices de Similaridad en Espacio Reducido ($d={}$)'.format(d), fontsize=22)
        plt.show()

# Función para visualizar los primeros autovectores
def plot_eigenvectors(VT, p, num_vectors, dataset):
    fig, axes = plt.subplots(1, num_vectors, figsize=(15, 6))
    for i in range(num_vectors):
        normalized_vector = (VT[i, :] - np.min(VT[i, :])) / (np.max(VT[i, :]) - np.min(VT[i, :]))
        im = axes[i].imshow(normalized_vector.reshape((p, p)), cmap='gray_r')
        axes[i].set_title(f'Autovector {i + 1}', fontsize=20)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    plt.subplots_adjust(wspace=0.5)
    plt.suptitle('Autovectores más representativos del dataset {}'.format(dataset), fontsize=28, y=0.85)
    plt.show()

# Función para calcular el error de reconstrucción utilizando la norma de Frobenius
def reconstruction_error_frobenius(X_original, X_reconstructed):
    return np.linalg.norm(X_original - X_reconstructed, 'fro') / np.linalg.norm(X_original, 'fro')

# Cargar imágenes de los datasets
dir_path = os.path.dirname(os.path.realpath(__file__))
images_1 = load_images_from_folder(os.path.join(dir_path, 'datasets_imgs_01'))
images_2 = load_images_from_folder(os.path.join(dir_path, 'datasets_imgs_02'))

# Parámetros
n_images, img_size = images_1.shape
p = int(np.sqrt(img_size))

# Visualizar imágenes originales y reconstruidas para distintas dimensiones
dimensions = [2, 10, 19, 50]
plot_reconstructed_images(images_1, dimensions)
plot_reconstructed_images_last_aves(images_1, dimensions)
plot_reconstructed_images_comparison(images_1, [19, 50])

# Definir sigma para la matriz de similaridad
sigma = 1000 # si el sigma es muy chiquito se muestran las 3 matrices iguales
# plot_similarity_matrices(images_1, dimensions, sigma)
plot_similarity_matrices(images_1, [2, 5, 10, 19], sigma)

# Visualizar los primeros autovectores para el dataset 1
U_50, s_50, VT_50 = svd(images_1, 50)
plot_eigenvectors(VT_50, p, num_vectors=4, dataset=1)

# Visualizar los primeros autovectores para el dataset 2
U_50_2, s_50_2, VT_50_2 = svd(images_2, 50)
plot_eigenvectors(VT_50_2, p, num_vectors=4, dataset=2)

# Encontrar el número mínimo de dimensiones d para dataset_imagenes2.zip con la norma de Frobenius
error_threshold = 0.10
min_d = None
errors_frobenius = []
max_d = 10

for d in range(1, max_d + 1):
    U, s, VT = svd(images_2, d)
    X_reduced_2 = U @ np.diag(s) @ VT
    error_frobenius = reconstruction_error_frobenius(images_2, X_reduced_2)
    errors_frobenius.append(error_frobenius)
    if error_frobenius <= error_threshold and min_d is None:
        min_d = d

print(f'Minimum number of dimensions to achieve < 10% error for dataset_imagenes2.zip: {min_d}')

# Aplanar las imágenes para aplicar SVD
images_2_flattened = images_2.reshape(images_2.shape[0], -1)
images_1_flattened = images_1.reshape(images_1.shape[0], -1)

# Utilizar la misma base de min_d dimensiones obtenida del dataset 2 para las imágenes del dataset 1
U, s, VT = svd(images_2_flattened, min_d)
X_reduced_1 = images_1_flattened @ VT.T
X_reconstructed_1 = X_reduced_1 @ VT

# Calcular el error de reconstrucción para dataset_imagenes1.zip
error_frobenius_1 = reconstruction_error_frobenius(images_1_flattened, X_reconstructed_1)
print(f'Reconstruction error for dataset_imagenes1.zip using d = {min_d}: {error_frobenius_1}')

# Graficar el error de reconstrucción en función de d
plt.figure(figsize=(12, 8))
plt.plot(range(1, max_d + 1), errors_frobenius, marker='o', linestyle='-', color='darkblue', label='Error de reconstrucción (Frobenius)')
plt.axhline(y=error_threshold, color='black', linestyle='--', label='Umbral de error del 10%')
plt.title('Error de Reconstrucción vs Número de Dimensiones', fontsize=20)
plt.xlabel(r'Número de dimensiones ($d$)', fontsize=16)
plt.ylabel('Error de reconstrucción (Norma de Frobenius)', fontsize=16)
plt.xticks(range(1, max_d + 1))
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Gráfico de error de reconstrucción para dataset_imagenes1.zip
errors = [reconstruction_error_frobenius(images_1_flattened, images_1_flattened @ svd(images_2_flattened, d)[2].T @ svd(images_2_flattened, d)[2]) for d in range(1, max_d + 1)]

# Calcular el error de reconstrucción para la dimensión d
d = min(10, U.shape[1])
U_d = U[:, :d]
s_d = s[:d]
VT_d = VT[:d, :]
sns.set_style("whitegrid")

X_reduced_1 = U_d @ np.diag(s_d) @ VT_d

# Inicializar matriz de errores para cada imagen
num_images = images_1.shape[0]
errors_per_image = np.zeros(num_images)

for i in range(num_images):
    original_image = images_1[i, :]
    reconstructed_image = X_reduced_1[i % X_reduced_1.shape[0], :]
    error_frobenius = np.linalg.norm(original_image - reconstructed_image) / np.linalg.norm(original_image)
    errors_per_image[i] = error_frobenius

# Graficar el error de reconstrucción para cada imagen
plt.figure(figsize=(15, 10))
plt.plot(range(1, num_images + 1), errors_per_image, marker='o', linestyle='-', color='darkblue')

plt.title('Error de Reconstrucción por Imagen', fontsize=20)
plt.xlabel('Índice de la imagen', fontsize=16)
plt.ylabel('Error de reconstrucción (Norma de Frobenius)', fontsize=16)
plt.xticks(range(1, num_images + 1))
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()

# Añadir una línea horizontal que muestra el error medio
mean_error = np.mean(errors_per_image)
plt.axhline(y=mean_error, color='black', linestyle='--', label=f'Error medio: {mean_error:.2f}')
plt.legend()

plt.show()

# Proyectar imágenes del dataset 1 usando la base del dataset 2
U_min_d, s_min_d, VT_min_d = svd(images_2, min_d)
X_reduced_1_using_2_basis = images_1 @ VT_min_d.T @ VT_min_d

# Visualizar la reconstrucción de imágenes proyectada en la base
fig, axes = plt.subplots(2, 9, figsize=(15, 6))
for i in range(9):
    # Imagen original
    axes[0, i].imshow(images_1[i].reshape((p, p)), cmap='gray')
    axes[0, i].set_title('Original', fontsize=16.5)
    axes[0, i].axis('off')

    # Imagen proyectada y reconstruida
    axes[1, i].imshow(X_reduced_1_using_2_basis[i].reshape((p, p)), cmap='gray')
    axes[1, i].set_title(r'Reconstruida ($d={}$)'.format(min_d), fontsize=10.5)
    axes[1, i].axis('off')

plt.suptitle('Imágenes originales y reconstruidas usando la base del dataset 2', fontsize=25)
plt.show()
# GRAFICO DE LAS IMAGENES DEL DATASET 2: YO LE ENSEÑÉ A MI MODELO 2 Y 8
# Visualizar las imágenes del dataset 2
fig, axes = plt.subplots(2, 4, figsize=(15, 6))
for i in range(8):
    row = i // 4
    col = i % 4
    axes[row, col].imshow(images_2[i].reshape((p, p)), cmap='gray')
    axes[row, col].set_title(f'Imagen {i+1}', fontsize=16.5)
    axes[row, col].axis('off')

plt.suptitle('Imágenes del dataset 2', fontsize=25)
plt.tight_layout()
plt.show()

# GRAFICO DE LAS IMAGENES DEL DATASET 1
fig, axes = plt.subplots(4, 5, figsize=(18, 14))
for i in range(19):
    row = i // 5
    col = i % 5
    axes[row, col].imshow(images_1[i].reshape((p, p)), cmap='gray')
    axes[row, col].set_title(f'Imagen {i+1}', fontsize=12)
    axes[row, col].axis('off')

# Desactivar el último subplot vacío
axes[3, 4].axis('off')

plt.suptitle('Imágenes del dataset 1', fontsize=22) 
plt.subplots_adjust(wspace=0.5, hspace=0.5) 
plt.show()

plot_similarity_matrices_sigma(images_1, dims=[2], sigmas=[10, 1000])


# AGREGAR AL APÉNDICE: desarrollar bien que notamos que para un sigma más grande notamos las diferencias, pero que para sigmas mas chiquitos las matrices son iguales y no se aprecian los clusters ¿por qué es así? ¿qué relación tiene el sigma?, entonces, explico este desarrollo en la parte de resultados y en apéndice agrego el gráfico de las matrices de similaridad con todos ceros alrededor y explico qué implica (similitudes entre imagenes)
# # el gráfico era para sigma = 10 por ejemplo

# para dimension 8 el error relativo entre cada dimension es menor al 10%. Quiero ver la matriz a de imagenes proyectada a eta base: A es la matriz de imagenes del dataset 1 de la parte 1, y hago y hago A*V*VT, con V de dimensión 8, es decir, ya hice SVD hasta dimensión 8 con el dataset 2, y ahora agarro el dataset uno y hago ese cálculo.

# En la reconstrucción del dataset 1 con base del dataset 2, SE MUESTRAN SOLO 2 Y 8 Y TIENE TOTAL SENTIDO, ES DECIR, SE INTENTAN RECONSTRUIR LOS VARIADOS NUMEROS DE LA PARTE 1 (9, 6, ETC) SOLO CON DOS Y OCHOS PORQUE ES CON LO QUE LE ENSEÑÉ AL MODELO!!