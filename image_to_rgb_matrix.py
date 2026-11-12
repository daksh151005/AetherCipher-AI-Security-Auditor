from PIL import Image
import numpy as np

def image_to_rgb_matrix(image_path):
    """
    Convert an image file to a numpy RGB matrix.
    """
    img = Image.open(image_path).convert('RGB')
    return np.array(img)

def crop_and_resize_to_square(image_path, target_size=None, save_path=None):
    """
    Center crop an image to 1:1 aspect ratio and optionally resize.
    """
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img_cropped = img.crop((left, top, left + min_dim, top + min_dim))
    if target_size and target_size < min_dim:
        img_cropped = img_cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)
    if save_path:
        img_cropped.save(save_path)
    return np.array(img_cropped)

