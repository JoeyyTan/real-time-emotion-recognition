import os
import cv2
import numpy as np
import joblib  # For loading the saved model
from skimage.feature import hog

# Load the pre-trained SVM model
print("Loading pre-trained SVM model...")
svm_model = joblib.load('gender_classifier_model.pkl')  # Load the saved SVM model

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define gender categories and colors
categories = ['Male', 'Female']
gender_colors = {
    'Male': (255, 0, 0),    # Blue
    'Female': (0, 255, 0)   # Green
}

# Define image size (used for resizing the face before processing)
IMG_SIZE = (128, 128)

# Function to preprocess the image
def preprocess_image(image):
    # Check if the image is already grayscale
    if len(image.shape) == 2:  # If the image has only one channel (grayscale)
        gray_image = image
    else:  # Convert to grayscale if it's not already
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    resized_image = cv2.resize(gray_image, IMG_SIZE)
    return resized_image

# Function to extract HOG features from the image
def extract_hog_features(image):
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

# Start video capture (use 1 for external webcam, or 0 for default)
cap = cv2.VideoCapture(1)  # Change to 0 for default webcam if needed

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Starting Real-Time Gender Detection...")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face region
            face = gray_frame[y:y+h, x:x+w]
            resized_face = preprocess_image(face)  # Resize and preprocess the face

            # Extract HOG features from the resized face
            features = extract_hog_features(resized_face)
            features = features.reshape(1, -1)  # SVM expects a 2D array (samples x features)

            # Predict the gender (0 = Male, 1 = Female)
            prediction = svm_model.predict(features)
            predicted_gender = categories[prediction[0]]

            # Get the color for the predicted gender
            color = gender_colors[predicted_gender]

            # Draw the bounding box and gender label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # Bounding box
            cv2.putText(frame, predicted_gender, (x, y - 10),  # Gender label
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Show the resulting frame with the bounding box and label
        cv2.imshow('Real-Time Gender Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
