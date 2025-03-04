import cv2
import numpy as np
from skimage.feature import hog
from skimage.transform import resize
from joblib import load
import os

# Load the trained vitiligo detection model
# Path to the trained model in the root folder
model_path = os.path.join(os.path.dirname(__file__), '..', 'trained_model.joblib')

# Debug: Print the model path
print("Model path:", model_path)

try:
    model = load(model_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{model_path}' does not exist. Please check the path.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")

# Load a pre-trained body part detection model (e.g., Haar cascade for face detection)
body_part_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess a region and extract HOG features
def preprocess_region(region):
    resized_region = resize(region, (64, 64))  # Resize to match training size
    fd, _ = hog(
        resized_region,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        channel_axis=-1
    )
    return fd.reshape(1, -1)  # Reshape for model input

# Function to detect body parts (e.g., face)
def detect_body_parts(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    body_parts = body_part_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return body_parts

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect body parts (e.g., face)
    body_parts = detect_body_parts(frame)

    # Loop over detected body parts
    for (x, y, w, h) in body_parts:
        # Crop the body part region
        body_part_region = frame[y:y + h, x:x + w]

        # Preprocess the region and extract HOG features
        processed_region = preprocess_region(body_part_region)

        # Make a prediction and get confidence score
        prediction = model.predict(processed_region)
        proba = model.predict_proba(processed_region)[0]  # Confidence scores

        # If the region is predicted as "Vitiligo"
        if prediction[0] == 1:  # Assuming 1 is the label for "Vitiligo"
            confidence = proba[1] * 100  # Confidence percentage for "Vitiligo"

            # Draw a green rectangle around the detected region
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the confidence percentage
            text = f"Vitiligo: {confidence:.2f}%"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Draw a red rectangle around the detected region (healthy)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Healthy", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time Vitiligo Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()