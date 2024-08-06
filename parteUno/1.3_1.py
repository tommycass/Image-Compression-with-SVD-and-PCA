import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir_path, 'datasets/dataset.csv')
pat_y = os.path.join(dir_path, 'datasets/y.txt')
labels = np.loadtxt(pat_y)
# Obtén la ruta del directorio del archivo actual

# Construye la ruta al archivo .csv

def processMatrix(matrix):
    matrix = pd.read_csv(path, header=None)
    # Elimino la primera fila y columna que son los nombres de las columnas y filas
    matrix = matrix.drop(0, axis=0)
    matrix = matrix.drop(0, axis=1)

    # Convierto la matriz a un array de numpy
    matrix = matrix.to_numpy()
    matrix = matrix - np.mean(matrix, axis=0) # esta cosa centra la matriz
    matrix = matrix / np.std(matrix, axis=0) # esta cosa normaliza la matriz

    return matrix

def reduceWithSVD(matrix, dimension):
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    U_reduced = u[:, :dimension]
    S_reduced = np.diag(s[:dimension])
    Vt_reduced = vt[:dimension, :]
    
    return U_reduced, S_reduced, Vt_reduced

def reduceWithPCA(matrix, dimension):
    u_reduced, s_reduced, vt_reduced = reduceWithSVD(matrix, dimension)
    return u_reduced @ s_reduced, vt_reduced

def leastSquares(pca_matrix, y, d):
    U_d, S_d, Vt_d = reduceWithSVD(pca_matrix, d)
    
    S_d_inv = np.diag([1/s if (s != 0) else 0 for s in np.diag(S_d)])
    X_pseudo_inv = Vt_d.T @ S_d_inv @ U_d.T
    
    beta = X_pseudo_inv @ y
    
    return beta

def calculate_error(pca_matrix, y, beta):
    # return np.linalg.norm(pca_matrix @ beta - y) ** 2
    return np.sqrt(np.mean((pca_matrix @ beta - y) ** 2))


def plot_prediction_error(X_train, y_train, X_test, y_test):
    errors = []
    dims = range(1, X_train.shape[1] + 1)
    approximation = None    
    for d in dims:
        X_train_pca, Vt_train_pca = reduceWithPCA(X_train, d)
        beta = leastSquares(X_train_pca, y_train, d)
        

        
        X_test_pca = X_test @ Vt_train_pca.T
        if False: # Esto es para debuggear
            print(f"""Dimension: {d}
                Tamaño de X_train_pca: {X_train_pca.shape}
                Tamaño de Vt_train_pca: {Vt_train_pca.shape}
                Tamaño de beta: {beta.shape}
                
                Tamaño de X_test_pca: {X_test_pca.shape}
                Tamaño de y_test: {y_test.shape}
                
                Tamaño de X_test_pca @ beta: {(X_test_pca @ beta).shape}
                
                Tamaño de vector final: {(X_test_pca @ beta - y_test).shape}""")
        if d == 2: approximation = X_test_pca @ beta
        error = calculate_error(X_test_pca, y_test, beta)
        errors.append(error)
        
    bestDimension = dims[np.argmin(errors)]
    print(f"La mejor dimensión es {bestDimension} con un error de {errors[bestDimension-1]}")
    
    
    # Crear una figura con dos subgráficos uno al lado del otro
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Primer gráfico
    axs[0].plot(dims, errors, color=(29/255, 73/255, 142/255))
    axs[0].set_title("Error variando la dimensión", fontsize=25)
    axs[0].set_xlabel("Dimensión", fontsize=16.5)
    axs[0].set_ylabel("Error en norma 2", fontsize=16.5)
    axs[0].grid()
    
    # Segundo gráfico
    axs[1].plot(range(len(y_test)), y_test, 'o', label='Real')
    axs[1].plot(range(len(approximation)), approximation, 'o', label='Aproximación', color="darkblue")
    axs[1].set_title("Valores reales vs aproximados", fontsize=25)
    axs[1].set_xlabel("Índice", fontsize=16.5)
    axs[1].set_ylabel("Valor", fontsize=16.5)
    axs[1].legend()
    axs[1].grid()
    
    # Ajustar el espaciado entre subgráficos
    plt.tight_layout()
    
    # Mostrar la figura
    plt.show()


def plotMatrixAndPlane(X_train, y_train, X_test, y_test):
    X_train_pca, Vt_train_pca = reduceWithPCA(X_train, 2)
    beta = leastSquares(X_train_pca, y_train, 2)
    
    X_test_pca = X_test @ Vt_train_pca.T
    
    # Combine train and test sets for visualization
    X_combined_pca = np.vstack((X_train_pca, X_test_pca))
    y_combined = np.concatenate((y_train, y_test))
    
    x_b = np.linspace(np.min(X_combined_pca[:, 0]), np.max(X_combined_pca[:, 0]), 10)
    y_b = np.linspace(np.min(X_combined_pca[:, 1]), np.max(X_combined_pca[:, 1]), 10)
    X_b, Y_b = np.meshgrid(x_b, y_b)
    Z_b = beta[0] * X_b + beta[1] * Y_b  
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_combined_pca[:, 0], X_combined_pca[:, 1], y_combined, cmap="viridis", c=y_combined, marker='o')
    ax.set_title("Aproximación de la función", fontsize=25)
    ax.set_xlabel("PCA1", fontsize=16.5)
    ax.set_ylabel("PCA2", fontsize=16.5)
    ax.plot_surface(X_b, Y_b, Z_b,alpha=0.5)
    plt.show()

def split_data(data, labels, train_ratio, shuffle = True):
    # Ensure the split is random but reproducible
    np.random.seed(42)
    indices = np.arange(data.shape[0])
    if shuffle: np.random.shuffle(indices)
    
    train_size = int(len(indices) * train_ratio)
    # Asigno primero a train y luego a test
    # train_indices = indices[:train_size]
    # test_indices = indices[train_size:]

    # Asigno primero a test y luego a train
    train_indices = indices[train_size:]
    test_indices = indices[:train_size]

    
    X_train = data[train_indices]
    X_test = data[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    
    return X_train, X_test, y_train, y_test

matrix = processMatrix(path)
qtrain_over_qtest = 0.3  # 1 = todos para train, 0 = todos para test
X_train, X_test, y_train, y_test = split_data(matrix, labels, qtrain_over_qtest)
# Calcular el error de predicción y graficar
plot_prediction_error(X_train, y_train, X_test, y_test)


labels = np.loadtxt(pat_y)
labels = (labels - np.mean(labels)) / np.std(labels)
qtrain_over_qtest = 0.3  # 1 = todos para train, 0 = todos para test
X_train, X_test, y_train, y_test = split_data(matrix, labels, qtrain_over_qtest)
plotMatrixAndPlane(X_train, y_train, X_test, y_test)



