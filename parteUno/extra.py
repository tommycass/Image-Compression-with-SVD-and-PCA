import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Función para cargar imágenes desde un directorio
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(folder, filename))
            images.append(np.array(img).flatten())
    return np.array(images)

# Funciones para aplicar los filtros
def apply_average_filter(image):
    return cv2.blur(image, (3, 3))

def apply_gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 1.0)

def apply_sobel_filter(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

def apply_canny_filter(image):
    return cv2.Canny(image, 100, 200)

# Cargar las imágenes desde los directorios
dir_path = os.path.dirname(os.path.realpath(__file__))
images_1 = load_images_from_folder(os.path.join(dir_path, 'datasets_imgs_01'))
images_2 = load_images_from_folder(os.path.join(dir_path, 'datasets_imgs_02'))

# Aplicar filtros a las imágenes cargadas y guardar los resultados
output_folder = os.path.join(dir_path, 'filtered_images')
os.makedirs(output_folder, exist_ok=True)

num_images_to_process = 5  # Número de imágenes a procesar y mostrar

for i in range(num_images_to_process):
    image = images_1[i]
    image_reshaped = image.reshape((28, 28))  # Suponiendo que las imágenes son de 28x28 píxeles

    # Aplicar los filtros
    average_filtered = apply_average_filter(image_reshaped)
    gaussian_filtered = apply_gaussian_filter(image_reshaped)
    sobel_filtered = apply_sobel_filter(image_reshaped)
    canny_filtered = apply_canny_filter(image_reshaped)

    # Guardar las imágenes resultantes
    cv2.imwrite(os.path.join(output_folder, f'imagen_{i}_promedios.png'), average_filtered)
    cv2.imwrite(os.path.join(output_folder, f'imagen_{i}_gaussiano.png'), gaussian_filtered)
    cv2.imwrite(os.path.join(output_folder, f'imagen_{i}_sobel.png'), sobel_filtered)
    cv2.imwrite(os.path.join(output_folder, f'imagen_{i}_canny.png'), canny_filtered)

# Visualizar las imágenes filtradas
plt.figure(figsize=(12, 10))

filters = [apply_average_filter, apply_gaussian_filter, apply_sobel_filter, apply_canny_filter]
titles = ['Filtro de Promedios', 'Filtro Gaussiano', 'Filtro de Sobel', 'Filtro de Canny']

# Seleccionar imágenes únicas para la visualización
num_unique_images = min(num_images_to_process, len(images_1))
selected_images = np.random.choice(len(images_1), num_unique_images, replace=False)

for i, filter_func in enumerate(filters):
    for j, image_index in enumerate(selected_images):
        image_sample = images_1[image_index].reshape((28, 28))
        filtered_image = filter_func(image_sample)

        plt.subplot(len(filters), num_unique_images, i * num_unique_images + j + 1)
        plt.imshow(filtered_image, cmap='gray')
        plt.title(f'{titles[i]}', fontsize=12)
        plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
