import cv2
import numpy as np
from skimage.feature import hog
from skimage.transform import resize
from joblib import load
import os

# Load the trained model for vitiligo classification
model_path = os.path.join(os.path.dirname(__file__), '..', 'trained_model.joblib')
model = load(model_path)  # Load the model from the project root

# Function to preprocess a region and extract HOG features
def preprocess_region(region):
    # Resize the region to match the input size used during training
    resized_region = resize(region, (64, 64))  # Adjust size as needed
    # Extract HOG features
    fd, _ = hog(
        resized_region,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        channel_axis=-1
    )
    return fd.reshape(1, -1)  # Reshape for model input

# Function to resize the image to fit within the screen
def resize_image_to_fit(image, max_width=800, max_height=600):
    """
    Resize the image to fit within the specified maximum width and height
    while maintaining the aspect ratio.
    """
    height, width = image.shape[:2]
    aspect_ratio = width / height

    # Calculate new dimensions
    if width > max_width or height > max_height:
        if aspect_ratio > 1:  # Landscape image
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        else:  # Portrait image
            new_height = max_height
            new_width = int(new_height * aspect_ratio)

        # Resize the image
        image = cv2.resize(image, (new_width, new_height))
    return image

# Function to detect vitiligo in an image
def detect_vitiligo(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    # Resize the image to fit within the screen
    image = resize_image_to_fit(image)

    # Convert image to RGB (HOG expects RGB images)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Sliding window parameters
    window_size = (64, 64)  # Size of the sliding window (adjust as needed)
    step_size = 32  # Step size for sliding window (adjust as needed)

    # Variables to track the best detection
    best_confidence = 0
    best_box = None

    # Loop over the image with a sliding window
    for (x, y, window) in sliding_window(image_rgb, window_size, step_size):
        # Preprocess the window
        processed_window = preprocess_region(window)

        # Make a prediction and get confidence score
        prediction = model.predict(processed_window)
        proba = model.predict_proba(processed_window)[0]  # Confidence scores

        # If the region is predicted as "Vitiligo"
        if prediction[0] == 1:  # Assuming 1 is the label for "Vitiligo"
            confidence = proba[1] * 100  # Confidence percentage for "Vitiligo"

            # Track the region with the highest confidence ≥ 90%
            if confidence >= 90 and confidence > best_confidence:
                best_confidence = confidence
                best_box = (x, y, window_size[0], window_size[1])

    # Display the result
    if best_box is not None:
        # Draw a green rectangle around the best detected region
        x, y, w, h = best_box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the confidence percentage
        text = f"Vitiligo: {best_confidence:.2f}%"
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        # Display the original image with a message
        cv2.putText(image, "No Vitiligo Found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the image in a resizable window
    cv2.namedWindow('Vitiligo Detection', cv2.WINDOW_NORMAL)
    cv2.imshow('Vitiligo Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Helper function for sliding window
def sliding_window(image, window_size, step_size):
    """Slide a window across the image."""
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# Main function
if __name__ == "__main__":
    # Path to the input image
    image_path = input("Enter the path to the image: ")

    # Detect vitiligo in the image
    detect_vitiligo(image_path)