import matplotlib.pyplot as plt

def plot_images(images, titles=None, n_cols=5):
    """Plot a grid of images."""
    n_rows = (len(images) + n_cols - 1) // n_cols
    plt.figure(figsize=(15, 10))
    for i, img in enumerate(images):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(img)
        if titles:
            plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()