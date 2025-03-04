from src.data_loader import load_dataset
from src.preprocess import preprocess_data
from src.feature_extraction import extract_hog_features
from src.model import prepare_data, train_model
from src.utils import save_data
from joblib import dump
import numpy as np

# Define the base folder containing 'healthy' and 'vitiligo' subfolders
base_folder = "C:\\Fythonn\\fyt\\Main\\Project_5\\data"  # Update this path if needed

# Load data
healthy_images, vitiligo_images = load_dataset(base_folder)

# Debugging: Print the number of images loaded
print(f"Number of healthy images: {len(healthy_images)}")
print(f"Number of vitiligo images: {len(vitiligo_images)}")

# Preprocess data
healthy_images, vitiligo_images = preprocess_data(healthy_images, vitiligo_images)

# Debugging: Print the number of images after preprocessing
print(f"Number of healthy images after preprocessing: {len(healthy_images)}")
print(f"Number of vitiligo images after preprocessing: {len(vitiligo_images)}")

# Extract HOG features
features_healthy = extract_hog_features(healthy_images)
features_vitiligo = extract_hog_features(vitiligo_images)

# Debugging: Print the number of feature vectors
print(f"Number of healthy feature vectors: {len(features_healthy)}")
print(f"Number of vitiligo feature vectors: {len(features_vitiligo)}")

# Combine features and labels
X = np.vstack((features_healthy, features_vitiligo))
y = [0] * len(features_healthy) + [1] * len(features_vitiligo)

# Debugging: Print the shapes of X and y
print(f"X shape: {X.shape}")  # Should be (num_samples, num_features)
print(f"y shape: {len(y)}")   # Should be (num_samples,)

# Prepare data for training
X_train, X_test, y_train, y_test = prepare_data(X, y)

# Debugging: Print the shapes of train/test sets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {len(y_train)}")
print(f"y_test shape: {len(y_test)}")

# Train model
model = train_model(X_train, X_test, y_train, y_test)

# Save processed data
save_data((X_train, X_test, y_train, y_test), "processed_data.pkl")

# Save the trained model
dump(model, "trained_model.joblib")
print("Trained model saved to 'trained_model.joblib'")