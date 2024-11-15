import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# Define the custom loss function
def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weighted_ce = cross_entropy * weights
        return K.sum(weighted_ce, axis=-1)

    return loss

# Class weights array (update this with your actual class weights)
class_weights_array = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # Replace with actual weights

# Path to the model
model_path = 'C:/Users/LENOVO/Downloads/real-time-emotion-recognition/cnn_trial.h5'

# Load the model without compilation
model = load_model(
    model_path,
    compile=False  # Avoid compiling during load
)

# Recompile the model with the custom loss function
model.compile(
    optimizer='adam',  # Replace with the optimizer used during training
    loss=weighted_categorical_crossentropy(class_weights_array),
    metrics=['accuracy']
)

print("Model loaded and recompiled successfully.")

# Define emotion categories and colors
categories = ['angry', 'fear', 'happy', 'neutral', 'sad']
emotion_colors = {
    'angry': (0, 0, 255),    # Red
    'fear': (255, 0, 0),     # Blue
    'happy': (0, 255, 0),    # Green
    'neutral': (255, 255, 0), # Cyan
    'sad': (255, 0, 255)     # Pink
}

# Define image size
IMG_SIZE = (48, 48)

# Start the webcam for real-time prediction
cap = cv2.VideoCapture(0)

print("Starting Real-Time Emotion Recognition...")
print("Press 'q' to exit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image. Exiting...")
        break

    # Detect faces using OpenCV's Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face = gray_frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face, IMG_SIZE)       # Resize to 48x48
        normalized_face = resized_face / 255.0         # Normalize pixel values
        reshaped_face = normalized_face.reshape(48, 48, 1)  # Add channel dimension

        # Convert grayscale to RGB by duplicating the grayscale values across 3 channels
        rgb_face = np.concatenate([reshaped_face] * 3, axis=-1)  # Shape: (48, 48, 3)
        rgb_face = np.expand_dims(rgb_face, axis=0)              # Add batch dimension

        # Make a prediction
        predictions = model.predict(rgb_face, verbose=0)
        predicted_class = np.argmax(predictions)
        predicted_emotion = categories[predicted_class]

        # Get the color for the predicted emotion
        color = emotion_colors[predicted_emotion]

        # Draw the bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show the video feed
    cv2.imshow('Real-Time Emotion Recognition', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
