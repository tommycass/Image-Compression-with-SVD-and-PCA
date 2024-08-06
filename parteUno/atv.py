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

# Centrar los datos con la media para calcular la matriz de similaridad
X_scaled = standard_scaler(X)

# Analizar autovalores y autovectores de X
U, s, VT = np.linalg.svd(X_scaled)

print("Las dimensiones más representativas son las correspondientes a los mayores valores singulares:")
print(s[:10])

# Graficar los valores singulares
plt.figure(figsize=(10, 6))
plt.bar(range(len(s)), s, color='blue', alpha=0.7)
plt.xlabel('Índice', fontsize=16.5)
plt.ylabel('Valor Singular', fontsize=16.5)
plt.title('Valores Singulares', fontsize=25)
plt.show()

# Calcular la varianza explicada por cada valor singular
explained_variance = (s ** 2) / np.sum(s ** 2)

# Calcular el porcentaje de representación de cada valor singular
explained_variance_ratio = explained_variance * 100

# Graficar el porcentaje de representación de cada valor singular
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(s) + 1), explained_variance_ratio, 'o-', linewidth=2, markersize=5)
plt.title('Porcentaje de representación de cada valor singular', fontsize=25)
plt.xlabel('Índice de valor singular', fontsize=16.5)
plt.ylabel('Porcentaje de varianza explicada (%)', fontsize=16.5)
plt.grid(True)
plt.show()

# Obtener los autovectores asociados con los dos primeros autovalores
first_eigenvector = VT[0, :]
second_eigenvector = VT[1, :]

# Crear una figura con dos subplots uno al lado del otro
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Graficar los pesos de cada componente en el primer autovector (valores absolutos)
ax1.bar(range(len(first_eigenvector)), np.abs(first_eigenvector), alpha=0.7)
ax1.set_xlabel('Componente', fontsize=16.5)
ax1.set_ylabel('Peso Absoluto', fontsize=16.5)
ax1.set_title('Primer autovector', fontsize=25)
ax1.grid(True)

# Graficar los pesos de cada componente en el segundo autovector (valores absolutos)
ax2.bar(range(len(second_eigenvector)), np.abs(second_eigenvector), color='darkblue', alpha=0.7)
ax2.set_xlabel('Componente', fontsize=16.5)
ax2.set_ylabel('Peso Absoluto', fontsize=16.5)
ax2.set_title('Segundo autovector', fontsize=25)
ax2.grid(True)

plt.show()
