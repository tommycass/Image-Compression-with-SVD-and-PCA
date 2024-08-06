import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Función para calcular distancias euclidianas
def euclidean_distances(X):
    m = X.shape[0]
    G = np.dot(X, X.T)
    H = np.tile(np.diag(G), (m, 1))
    return np.sqrt(H + H.T - 2 * G)

# Escalar los datos
X_scaled = standard_scaler(X)

# Realizar PCA para d = 2, 6, 10, y p
#dimensions = [2, 6, 10, X.shape[1]]
#for d in dimensions:
#    X_pca = pca(X_scaled, n_components=d)
#
#    # Calcular matriz de similaridad
#    sigma_similarity = 15.0  # Ajustar según sea necesario
#    S_pca = np.exp(-euclidean_distances(X_pca)**2 / (2 * sigma_similarity**2))
#
#    # Graficar matriz de similaridad
#    plt.figure(figsize=(10, 8))
#    sns.heatmap(S_pca, cmap='viridis', xticklabels=False, yticklabels=False, cbar_kws={'label': 'Similarity'})
#    plt.title(f'Similarity Matrix in PCA Space (d={d})')
#    plt.show()

# Ajustar sigma para la matriz de similaridad
sigma_similarity = 0.001  # Ajustar según sea necesario
# Calcular matriz de similaridad con el nuevo sigma
S_original = np.exp(-euclidean_distances(X_scaled)**2 / (2 * sigma_similarity**2))
# Graficar la matriz de similaridad en el espacio original
plt.figure(figsize=(10, 8))
sns.heatmap(S_original, cmap='viridis', xticklabels=False, yticklabels=False, cbar_kws={'label': 'Similaridad'})
plt.title('Matriz de Similaridad en Espacio Original', fontsize=25) 
plt.xlabel('Features', fontsize=16.5)
plt.ylabel('Muestras', fontsize=16.5)
plt.show()
