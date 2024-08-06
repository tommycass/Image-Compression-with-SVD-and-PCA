import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Obtén la ruta del directorio del archivo actual
dir_path = os.path.dirname(os.path.realpath(__file__))

# Construye la ruta al archivo .csv
csv_path = os.path.join(dir_path, 'datasets/dataset.csv')

# Carga el dataset
data = pd.read_csv(csv_path, header=None, skiprows=1, usecols=range(1, 205))
X = data.values

# Función para escalar los datos
def standard_scaler(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

# Función para realizar PCA
def pca(X, n_components):
    # Centrar los datos
    X_centered = X - np.mean(X, axis=0)
    # Calcular la matriz de covarianza
    covariance_matrix = np.cov(X_centered, rowvar=False)
    # Calcular los valores y vectores propios
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # Ordenar los vectores propios por los valores propios en orden descendente
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    # Seleccionar los primeros n_components vectores propios
    eigenvectors_subset = sorted_eigenvectors[:, :n_components]
    # Transformar los datos
    X_reduced = np.dot(X_centered, eigenvectors_subset)
    return X_reduced

# Escalar los datos
X_scaled = standard_scaler(X)

# Realizar PCA con 3 componentes para la visualización en 3D
X_pca_3d = pca(X_scaled, n_components=3)

# Realizar PCA con 2 componentes para la visualización en 2D
X_pca_2d = pca(X_scaled, n_components=2)

# Crear una figura con 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Graficar PCA en 2D en el primer subplot
sc2 = axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=X_pca_2d[:, 1], cmap='viridis')
axes[0].set_title('PCA en 2D', fontsize=25)
axes[0].set_xlabel('PCA1', fontsize=16.5)
axes[0].set_ylabel('PCA2', fontsize=16.5)
axes[0].tick_params(axis='both', which='major', labelsize=12)
# Ocultar los valores de los ejes
axes[0].set_xticks([])
axes[0].set_yticks([])

# Graficar PCA en 3D en el segundo subplot
ax = fig.add_subplot(122, projection='3d')
sc = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=X_pca_3d[:, 2], cmap='viridis')
ax.set_title('PCA en 3D', fontsize=25)
ax.set_xlabel('PCA1', fontsize=16.5)
ax.set_ylabel('PCA2', fontsize=16.5)
ax.set_zlabel('PCA3', fontsize=16.5)
ax.tick_params(axis='both', which='major', labelsize=12)
# Ocultar los valores de los ejes
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Ajustar el layout y mostrar el gráfico
plt.tight_layout()
plt.show()
