import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
data_dir = 'C:\\Fythonn\\fyt\\Main\\Project_5\\data'  # Path to your data folder
train_dir = os.path.join(data_dir, 'train')  # Training data folder
val_dir = os.path.join(data_dir, 'validation')  # Validation data folder

# Create train and validation directories
os.makedirs(os.path.join(train_dir, 'vitiligo'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'healthy'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'vitiligo'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'healthy'), exist_ok=True)

# Split ratio (80% training, 20% validation)
split_ratio = 0.2

# Split data for each class
for class_name in ['vitiligo', 'healthy']:
    class_dir = os.path.join(data_dir, class_name)  # Original class folder
    train_class_dir = os.path.join(train_dir, class_name)  # Training class folder
    val_class_dir = os.path.join(val_dir, class_name)  # Validation class folder

    # List all files in the class directory (only image files)
    files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    if len(files) == 0:
        print(f"No files found in {class_dir}. Skipping...")
        continue  # Skip this directory

    # Split files into training and validation sets
    train_files, val_files = train_test_split(files, test_size=split_ratio, random_state=42)

    # Move training files to the train folder
    for file in train_files:
        shutil.move(os.path.join(class_dir, file), os.path.join(train_class_dir, file))

    # Move validation files to the validation folder
    for file in val_files:
        shutil.move(os.path.join(class_dir, file), os.path.join(val_class_dir, file))

print("Data splitting and moving completed.")