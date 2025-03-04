from skimage.feature import hog
from skimage import exposure
import numpy as np

def extract_hog_features(images):
    """
    Extract HOG features from a list of images.

    Args:
        images (list): List of images (as NumPy arrays).

    Returns:
        np.array: Array of HOG feature vectors.
    """
    features = []
    for img in images:
        # Extract HOG features
        fd, hog_image = hog(
            img,
            pixels_per_cell=(8, 8),  # Size of each cell
            cells_per_block=(2, 2),  # Number of cells in each block
            visualize=True,          # Return the HOG image for visualization
            channel_axis=-1,         # Handle multi-channel (color) images
        )
        features.append(fd)  # Append the HOG feature vector
    return np.array(features)  # Convert to a NumPy array