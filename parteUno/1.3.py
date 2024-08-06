import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

def load_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    csv_path = os.path.join(dir_path, 'datasets/dataset.csv')
    csv_path_y = os.path.join(dir_path, 'datasets/y.txt')
    X = pd.read_csv(csv_path, skiprows=1, usecols=range(1, 205)).values
    y = pd.read_csv(csv_path_y).values.flatten()
    return X, y

def normalize_dataset(dataset):
    return (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)

def pseudo_inverse(S_d):
    S_d_inv = np.zeros_like(S_d)
    
    for i in range(len(S_d)):
        if S_d[i, i] != 0:
            S_d_inv[i, i] = 1 / S_d[i, i]
    
    return S_d_inv

def generate_pca(X, d):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    Vt_d = Vt[:d, :]
    
    X_pca = U_d @ S_d @ Vt_d
    
    return X_pca, U_d, S_d, Vt_d

def svd_least_squares_PCA(X, y, d):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    Vt_d = Vt[:d, :]
    
    X_pseudo_inv = Vt_d.T @ pseudo_inverse(S_d) @ U_d.T
    
    beta = X_pseudo_inv @ y
    
    # error norma 2 al cuadrado
    error = np.linalg.norm(X @ beta - y) ** 2
    
    return X_pseudo_inv, beta, error

def plot_prediction_errors(X, y):
    errors = []
    dims = range(1, 11)
    
    for d in dims:
        X_pca, U_d, S_d, Vt_d = generate_pca(X, d)
        
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.7, random_state=42)
        
        X_pseudo_inv, beta, _ = svd_least_squares_PCA(X_train, y_train, d)
        
        # Predecir en el conjunto de prueba y calcular el RECM
        y_pred = X_test @ beta
        recm = np.sqrt(np.mean((y_test - y_pred) ** 2))
        errors.append(recm)
       
    best_dimension = dims[np.argmin(errors)]   
    
    plt.figure(figsize=(12, 6))
    plt.plot(dims, errors, 'o-', markersize=2.5, color="darkcyan", linewidth=2)
    plt.xlabel('Dimensiones')
    plt.ylabel('Error de predicción (RECM)')
    plt.title('Error de predicción para diferentes dimensiones')
    plt.grid(True)
    plt.show()
    
    print(f"La mejor dimensión es {best_dimension} con un RECM de {errors[best_dimension-1]}")
    
    return best_dimension

def plot_beta_weights(beta):
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(beta) + 1), beta)
    plt.title('Pesos del Vector β en el Espacio Original')
    plt.xlabel('Dimensiones Originales')
    plt.ylabel('Pesos de β')
    plt.grid(True)
    plt.show()


def main():
    X, y = load_data()
    X = normalize_dataset(X)    
    best_dimension = plot_prediction_errors(X, y)
    X_pca, _, _, _ = generate_pca(X, best_dimension)
    _, beta, _ = svd_least_squares_PCA(X_pca, y, best_dimension)
    plot_beta_weights(beta)
    
if __name__ == "__main__":
    main()
