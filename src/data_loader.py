import os
from PIL import Image

def load_images_from_folder(folder_path):
    """
    Load images from a folder.

    Args:
        folder_path (str): Path to the folder containing images.

    Returns:
        list: List of images (as PIL Image objects).
    """
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(img_path).convert('RGB')  # Ensure images are in RGB format
            images.append(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    return images

def load_dataset(base_folder):
    """
    Load healthy and vitiligo images from their respective folders.

    Args:
        base_folder (str): Path to the base folder containing 'healthy' and 'vitiligo' subfolders.

    Returns:
        tuple: (healthy_images, vitiligo_images) where each is a list of PIL Image objects.
    """
    healthy_folder = os.path.join(base_folder, "healthy Skin")
    vitiligo_folder = os.path.join(base_folder, "vitiligo")
    
    healthy_images = load_images_from_folder(healthy_folder)
    vitiligo_images = load_images_from_folder(vitiligo_folder)
    
    return healthy_images, vitiligo_images