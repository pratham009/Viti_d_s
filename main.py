import cv2
import numpy as np
import tensorflow as tf
from skimage.transform import resize

# Load the trained CNN model
model = tf.keras.models.load_model('vitiligo_detection_cnn.h5')

# Function to preprocess a region for the CNN model
def preprocess_region(region):
    resized_region = resize(region, (64, 64))  # Resize to match model input size
    resized_region = resized_region / 255.0    # Normalize pixel values to [0, 1]
    return np.expand_dims(resized_region, axis=0)  # Add batch dimension

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (CNN expects RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess the frame for the CNN model
    processed_frame = preprocess_region(frame_rgb)

    # Make a prediction
    prediction = model.predict(processed_frame)
    confidence = prediction[0][0] * 100  # Confidence percentage

    # Determine the label
    label = "Vitiligo" if confidence >= 50 else "Healthy"

    # Display the label and confidence
    text = f"{label}: {confidence:.2f}%"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time Vitiligo Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
