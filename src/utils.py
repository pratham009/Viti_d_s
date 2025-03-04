import pickle

def save_data(data, file_path):
    """
    Save data to a file.

    Args:
        data: Data to save.
        file_path (str): Path to the file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_data(file_path):
    """
    Load data from a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        Data loaded from the file.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)