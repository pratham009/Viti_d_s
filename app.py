from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
from skimage.feature import hog
from skimage.transform import resize
from joblib import load
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'trained_model.joblib')
model = load(model_path)  # Load the model

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

# Function to detect vitiligo in an image
def detect_vitiligo(image):
    # Convert image to RGB (HOG expects RGB images)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Sliding window parameters
    window_size = (64, 64)  # Size of the sliding window
    step_size = 32  # Step size for sliding window

    # Variables to track the best detection
    best_confidence = 0
    best_box = None
    best_label = "Healthy"

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

            # Track the region with the highest confidence
            if confidence > best_confidence:
                best_confidence = confidence
                best_box = (x, y, window_size[0], window_size[1])
                best_label = "Vitiligo" if confidence >= 90 else "Healthy"

    # Draw the best detection on the frame
    if best_box is not None:
        x, y, w, h = best_box
        color = (0, 255, 0) if best_label == "Vitiligo" else (0, 0, 255)  # Green for Vitiligo, Red for Healthy
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # Display the label and confidence percentage
        text = f"{best_label}: {best_confidence:.2f}%"
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

# Helper function for sliding window
def sliding_window(image, window_size, step_size):
    """Slide a window across the image."""
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# Route to handle image upload and processing
@app.route('/process', methods=['POST'])
def process():
    # Get the image file from the request
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Detect vitiligo in the image
    processed_image = detect_vitiligo(image)

    # Encode the processed image as JPEG
    _, buffer = cv2.imencode('.jpg', processed_image)
    response = buffer.tobytes()

    # Return the processed image as a response
    return Response(response, mimetype='image/jpeg')

# Serve the HTML page
@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vitiligo Detection</title>
    </head>
    <body>
        <h1>Vitiligo Detection</h1>
        <video id="video" autoplay></video>
        <canvas id="canvas" style="display:none;"></canvas>
        <img id="output" />
        <button id="capture">Capture</button>

        <script>
                    const video = document.getElementById('video');
                    const canvas = document.getElementById('canvas');
                    const output = document.getElementById('output');
                    const captureButton = document.getElementById('capture');

                    // Access the phone's camera
                    navigator.mediaDevices.getUserMedia({ video: true })
                        .then(stream => {
                            video.srcObject = stream;
                            console.log("Camera access granted.");
                        })
                        .catch(err => {
                            console.error('Error accessing camera:', err);
                        });

                    // Capture image and send to server
                    captureButton.addEventListener('click', () => {
                        const context = canvas.getContext('2d');
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        context.drawImage(video, 0, 0, canvas.width, canvas.height);
                        console.log("Frame captured on canvas.");

                        // Convert canvas image to JPEG and send to server
                        canvas.toBlob(blob => {
                            console.log("Blob created:", blob);
                            const formData = new FormData();
                            formData.append('image', blob, 'frame.jpg');

                            fetch('/process', {
                                method: 'POST',
                                body: formData
                            })
                            .then(response => {
                                if (!response.ok) {
                                    throw new Error('Network response was not ok');
                                }
                                return response.blob();
                            })
                            .then(blob => {
                                console.log("Processed image received from server.");
                                const url = URL.createObjectURL(blob);
                                output.src = url;
                            })
                            .catch(err => {
                                console.error('Error processing image:', err);
                            });
                        }, 'image/jpeg', 0.9);
                    });
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)