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
data = pd.read_csv(csv_path, skiprows=1, usecols=range(1, 205))
X = data.values


# Definir funciones de escalamiento
def standard_scaler(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

def min_max_scaler(S):
    S_min = np.min(S)
    S_max = np.max(S)
    return (S - S_min) / (S_max - S_min)

# Centrar los datos con la media para calcular la matriz de similaridad
X_scaled = standard_scaler(X)
# Definir funciones de escalamiento

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

# Función para calcular la matriz de similaridad basada en producto punto
def linear_similarity_matrix(X):
    return np.dot(X, X.T)

# Función para normalizar la matriz de similaridad entre 0 y 1
def normalize_matrix(S):
    return min_max_scaler(S)

X_pca_2d = pca(X_scaled, n_components=10)

# Calcular matriz de similaridad en el espacio original usando producto punto
S_linear = linear_similarity_matrix(X_pca_2d)
# Normalizar la matriz de similaridad en el espacio original
S_linear_normalized = normalize_matrix(S_linear)

# Función para calcular distancias euclidianas
def euclidean_distances(X):
    m = X.shape[0]
    G = np.dot(X, X.T)
    H = np.tile(np.diag(G), (m, 1))
    return np.sqrt(H + H.T - 2 * G)

# Ajustar sigma para la matriz de similaridad
sigma_similarity = 10  # Ajustar según sea necesario
# Calcular matriz de similaridad con el nuevo sigma usando distancia euclidiana
S_euclidean = np.exp(-euclidean_distances(X_pca_2d)**2 / (2 * sigma_similarity**2))

# Crear los subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Graficar matriz de similaridad lineal normalizada en el primer subplot
sns.heatmap(S_linear_normalized, cmap='viridis', xticklabels=False, yticklabels=False, ax=axes[0])
axes[0].set_title('Distancia Lineal', fontsize=25)
axes[0].set_xlabel("Muestras", fontsize=16.5)
axes[0].set_ylabel("Features", fontsize=16.5)

# Graficar matriz de similaridad euclidiana en el segundo subplot
sns.heatmap(S_euclidean, cmap='viridis', xticklabels=False, yticklabels=False, ax=axes[1])
axes[1].set_title('Kernel Gaussiano', fontsize=25)
axes[1].set_xlabel('Muestras', fontsize=16.5)
axes[1].set_ylabel('Features', fontsize=16.5)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()
