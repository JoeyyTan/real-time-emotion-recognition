import cv2
import numpy as np
from skimage.feature import hog
import joblib

# Load the pre-trained SVM model
svm = joblib.load('gender_classifier_model.pkl')

# Initialize the face detector (using Haar cascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess images (same as training)
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (128, 128))
    return resized_image

# Function to extract HOG features (same as training)
def extract_hog_features(image):
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

# Open webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) - the face
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess the face ROI
        processed_face = preprocess_image(face_roi)
        
        # Extract HOG features
        face_features = extract_hog_features(processed_face)
        
        # Reshape for prediction
        face_features = face_features.reshape(1, -1)
        
        # Predict gender (0: Male, 1: Female)
        prediction = svm.predict(face_features)[0]
        gender_label = "Male" if prediction == 0 else "Female"
        
        # Display the prediction on the frame
        color = (255, 0, 0) if gender_label == "Male" else (255, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, gender_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Show the frame with detected faces and gender labels
    cv2.imshow('Gender Detection', frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
