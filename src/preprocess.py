from PIL import Image
import numpy as np

def resize_images(images, target_size=(64, 64)):
    """
    Resize images to a target size.

    Args:
        images (list): List of images (as PIL Image objects).
        target_size (tuple): Target size (width, height).

    Returns:
        list: List of resized images.
    """
    return [img.resize(target_size) for img in images]

def normalize_images(images):
    """
    Normalize pixel values to [0, 1].

    Args:
        images (list): List of images (as PIL Image objects).

    Returns:
        list: List of normalized images.
    """
    return [np.array(img) / 255.0 for img in images]

def preprocess_data(healthy_images, vitiligo_images, target_size=(64, 64)):
    """
    Preprocess the dataset.

    Args:
        healthy_images (list): List of healthy images.
        vitiligo_images (list): List of vitiligo images.
        target_size (tuple): Target size for resizing.

    Returns:
        tuple: (healthy_images, vitiligo_images) where each is a list of preprocessed images.
    """
    # Resize images
    healthy_images = resize_images(healthy_images, target_size)
    vitiligo_images = resize_images(vitiligo_images, target_size)
    
    # Normalize images
    healthy_images = normalize_images(healthy_images)
    vitiligo_images = normalize_images(vitiligo_images)
    
    # Return the preprocessed images
    return healthy_images, vitiligo_images