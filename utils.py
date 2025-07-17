import os
import numpy as np
import skimage.io
import skimage.transform

# === Config ===
IMAGE_REDUCED_SIZE = 64
IMAGE_DIR = "images"
FOLDERS = ['letters', 'numbers', 'flags', 'animals']
SESSION_SIZE = 15 

def load_dataset(folders):
    filenames, labels, categories = [], [], []
    for folder in folders:
        folder_path = os.path.join(IMAGE_DIR, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                full_path = os.path.join(folder_path, filename)
                filenames.append(full_path)
                labels.append(os.path.splitext(filename)[0])  # e.g. "A", "3", "dog"
                categories.append(folder)  # e.g. "letters"
    return filenames, labels, categories

def load_images(filenames):
    images = []
    for filename in filenames:
        image = skimage.io.imread(filename)
        image = skimage.transform.resize(image, (IMAGE_REDUCED_SIZE, IMAGE_REDUCED_SIZE))
        if image.ndim == 2:  # (64, 64)
            image = np.stack([image]*3, axis=-1)
        if image.shape[-1] == 1:  # (64, 64, 1)
            image = np.repeat(image, 3, axis=-1)
        if image.shape[-1] > 3:  # (64, 64, 4) RGBA
            image = image[..., :3]
        images.append(image)
    return np.array(images)
