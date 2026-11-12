import numpy as np
from PIL import Image

def arnold_cat_map(image_matrix, iterations=1):
    """
    Apply Arnold Cat Map to scramble the image matrix.
    image_matrix: numpy array of shape (N, N, 3)
    iterations: number of times to apply the map
    """
    N = image_matrix.shape[0]
    result = np.copy(image_matrix)
    x, y = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    new_x = (x + y) % N
    new_y = (x + 2 * y) % N
    for _ in range(iterations):
        temp = np.zeros_like(result)
        temp[new_x, new_y] = result
        result = temp
    return result

def inverse_arnold_cat_map(image_matrix, iterations=1):
    """
    Apply inverse Arnold Cat Map to descramble the image matrix.
    """
    N = image_matrix.shape[0]
    result = np.copy(image_matrix)
    x, y = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    new_x = (2 * x - y) % N
    new_y = (-x + y) % N
    for _ in range(iterations):
        temp = np.zeros_like(result)
        temp[new_x, new_y] = result
        result = temp
    return result

def logistic_map_scramble(image_matrix, x, r):
    """
    Scramble the image matrix using logistic map.
    x: initial value (0 < x < 1)
    r: parameter (usually between 3.57 and 4)
    """
    N = image_matrix.shape[0]
    total_pixels = N * N
    sequence = np.empty(total_pixels, dtype=np.float64)
    val = float(x)
    r_val = float(r)
    for i in range(total_pixels):
        val = r_val * val * (1.0 - val)
        sequence[i] = val
    indices = np.argsort(sequence)
    flat = image_matrix.reshape((total_pixels, 3))
    scrambled = flat[indices]
    return scrambled.reshape((N, N, 3))

def logistic_map_descramble(image_matrix, x, r):
    """
    Descramble the image matrix using logistic map.
    """
    N = image_matrix.shape[0]
    total_pixels = N * N
    sequence = np.empty(total_pixels, dtype=np.float64)
    val = float(x)
    r_val = float(r)
    for i in range(total_pixels):
        val = r_val * val * (1.0 - val)
        sequence[i] = val
    indices = np.argsort(sequence)
    flat = image_matrix.reshape((total_pixels, 3))
    descrambled = np.empty_like(flat)
    descrambled[indices] = flat
    return descrambled.reshape((N, N, 3))

def save_image_from_matrix(image_matrix, path):
    """
    Save a numpy image matrix as an image file.
    """
    img = Image.fromarray(np.uint8(image_matrix))
    img.save(path)

