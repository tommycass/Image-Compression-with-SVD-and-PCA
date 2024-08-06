import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Definición de funciones personalizadas

def custom_pca(X, n_components=2):
    # Centrar los datos
    X_centered = X - np.mean(X, axis=0)
    # Calcular la matriz de covarianza
    covariance_matrix = np.cov(X_centered, rowvar=False)
    # Calcular los valores y vectores propios
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # Ordenar los vectores propios por los valores propios más grandes
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    # Seleccionar los primeros n_components vectores propios
    eigenvector_subset = sorted_eigenvectors[:, :n_components]
    # Transformar los datos
    X_reduced = np.dot(eigenvector_subset.T, X_centered.T).T
    return X_reduced

def custom_euclidean_distances(X):
    sum_X = np.sum(np.square(X), axis=1)
    distances = np.sqrt(np.maximum(sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * np.dot(X, X.T), 0))
    return distances

def custom_standard_scaler(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

# Obtén la ruta del directorio del archivo actual
dir_path = os.path.dirname(os.path.realpath(__file__))

# Construye la ruta al archivo .csv
csv_path = os.path.join(dir_path, 'datasets/dataset.csv')

# Carga el dataset
data = pd.read_csv(csv_path, header=None, skiprows=1, usecols=range(1, 205))
X = data.values

# Centrar los datos con la media para calcular la matriz de similaridad
X_scaled = custom_standard_scaler(X)

# Realizar PCA para d = 2
X_pca = custom_pca(X_scaled, n_components=2)

# Definir los valores de sigma_similarity a probar (quita uno de los valores)
sigma_values = [0.1, 1, 10]

# Crear una figura con subplots 1x3
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for i, sigma in enumerate(sigma_values):
    # Calcular matriz de similaridad
    S_pca = np.exp(-custom_euclidean_distances(X_pca)**2 / (2 * sigma**2))

    # Graficar matriz de similaridad
    ax = axes[i]
    sns.heatmap(S_pca, cmap='viridis', xticklabels=False, yticklabels=False, ax=ax)
    ax.set_title(f'σ={sigma}', fontsize=25)
    ax.set_xlabel('Features', fontsize=16.5)
    ax.set_ylabel('Muestras', fontsize=16.5)

# Ajustar el layout y mostrar el gráfico
plt.tight_layout()
plt.show()
