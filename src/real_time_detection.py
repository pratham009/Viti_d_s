import cv2
import numpy as np
import tensorflow as tf

# Load the trained CNN model
model = tf.keras.models.load_model('vitiligo_detection_cnn.h5')

# Function to preprocess a frame for the model
def preprocess_frame(frame):
    """Preprocess a frame for the model."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    resized_frame = cv2.resize(frame_rgb, (64, 64))  # Resize to 64x64
    normalized_frame = resized_frame / 255.0  # Normalize pixel values
    return np.expand_dims(normalized_frame, axis=0)  # Add batch dimension

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Confidence threshold (adjust as needed)
confidence_threshold = 0.7

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Make a prediction
    prediction = model.predict(processed_frame)
    confidence = prediction[0][0]  # Confidence score

    # Determine the label based on the confidence threshold
    if confidence >= confidence_threshold:
        label = "Vitiligo"
        color = (0, 255, 0)  # Green for vitiligo
    else:
        label = "Healthy"
        color = (0, 0, 255)  # Red for healthy

    # Display the label and confidence
    text = f"{label}: {confidence * 100:.2f}%"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the resulting frame
    cv2.imshow('Real-Time Vitiligo Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()