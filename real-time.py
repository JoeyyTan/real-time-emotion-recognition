import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.feature import hog  
import joblib

# Function to overlay the gender icon on top of the face and to the right
def overlay_icon(frame, icon, x, y, w, h):
    # Get original icon dimensions
    ih, iw = icon.shape[:2]

    # Calculate the new dimensions while maintaining the aspect ratio (1/4 size of the face)
    new_h = h // 4  # Icon height is 1/4 of face height
    new_w = int(new_h * iw / ih)  # Resize the width while maintaining aspect ratio

    # Resize the icon to the new dimensions
    resized_icon = cv2.resize(icon, (new_w, new_h))

    # Calculate the position to place the icon on the right side, but above the face
    x_offset = x + w - new_w - 10  # 10px margin from the right edge of the face
    y_offset = y - new_h - 10  # Place the icon above the face with a 10px margin

    # Split the icon into color and alpha channels
    icon_rgb = resized_icon[:, :, :3]
    icon_alpha = resized_icon[:, :, 3]

    # Ensure the icon is within frame boundaries
    x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + new_w)
    y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + new_h)
    icon_rgb = icon_rgb[y1 - y_offset:y2 - y_offset, x1 - x_offset:x2 - x_offset]
    icon_alpha = icon_alpha[y1 - y_offset:y2 - y_offset, x1 - x_offset:x2 - x_offset]

    # Extract the region of interest (ROI) from the frame
    roi = frame[y1:y2, x1:x2]

    # Blend the icon with the ROI
    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1 - icon_alpha / 255.0) + icon_rgb[:, :, c] * (icon_alpha / 255.0)

    # Place the blended ROI back into the frame
    frame[y1:y2, x1:x2] = roi



# Function to overlay emotion filter on the face
def overlay_filter(frame, filter_img, x, y, w, h):
    fh, fw = filter_img.shape[:2]
    aspect_ratio = fw / fh
    if w / h > aspect_ratio:
        new_h = h
        new_w = int(h * aspect_ratio)
    else:
        new_w = w
        new_h = int(w / aspect_ratio)

    filter_resized = cv2.resize(filter_img, (new_w, new_h))
    x_offset = x + (w - new_w) // 2
    y_offset = y + (h - new_h) // 2

    filter_rgb = filter_resized[:, :, :3]
    filter_alpha = filter_resized[:, :, 3]

    x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + new_w)
    y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + new_h)
    filter_rgb = filter_rgb[y1 - y_offset:y2 - y_offset, x1 - x_offset:x2 - x_offset]
    filter_alpha = filter_alpha[y1 - y_offset:y2 - y_offset, x1 - x_offset:x2 - x_offset]

    roi = frame[y1:y2, x1:x2]
    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1 - filter_alpha / 255.0) + filter_rgb[:, :, c] * (filter_alpha / 255.0)
    frame[y1:y2, x1:x2] = roi

recent_predictions = []
window_size = 10  # Adjust based on your frame rate and desired smoothness

def smooth_prediction(new_prediction):
    recent_predictions.append(new_prediction)
    if len(recent_predictions) > window_size:
        recent_predictions.pop(0)  # Remove oldest prediction

    # Return the most common prediction in the window
    return max(set(recent_predictions), key=recent_predictions.count)



# Load the emotion model
model_path = 'C:/Users/LENOVO/Downloads/real-time-emotion-recognition/emotion_model.h5'
model = load_model(model_path, compile=False)

categories = ['angry', 'fear', 'happy', 'neutral', 'sad']

# Load the gender detection model
gender_model_path = 'C:/Users/LENOVO/Downloads/real-time-emotion-recognition/gender_classifier_model.pkl'
gender_model = joblib.load(gender_model_path)

# Define paths for emotion filters
filter_paths = {
    'angry': 'C:/Users/LENOVO/Downloads/real-time-emotion-recognition/filters/path_to_angry_filter.png',
    'fear': 'C:/Users/LENOVO/Downloads/real-time-emotion-recognition/filters/path_to_fear_filter.png',
    'happy': 'C:/Users/LENOVO/Downloads/real-time-emotion-recognition/filters/path_to_happy_filter.png',
    'neutral': 'C:/Users/LENOVO/Downloads/real-time-emotion-recognition/filters/path_to_neutral_filter.png',
    'sad': 'C:/Users/LENOVO/Downloads/real-time-emotion-recognition/filters/path_to_sad_filter.png',
}

filters = {emotion: cv2.imread(path, cv2.IMREAD_UNCHANGED) for emotion, path in filter_paths.items()}

# Load gender icons as images with alpha channels
male_icon = cv2.imread('C:/Users/LENOVO/Downloads/real-time-emotion-recognition/filters/male_icon.png', cv2.IMREAD_UNCHANGED)
female_icon = cv2.imread('C:/Users/LENOVO/Downloads/real-time-emotion-recognition/filters/female_icon.png', cv2.IMREAD_UNCHANGED)


# Function to preprocess face for SVM (gender detection)
def preprocess_for_gender_detection(face):
    face_resized = cv2.resize(face, (128, 128))
    face_hog, _ = hog(face_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return face_hog.reshape(1, -1)

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise and improve face detection
    blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(blurred_frame, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))

    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]  # Use original grayscale face (not blurred) for recognition
        resized_face = cv2.resize(face, (48, 48))
        normalized_face = resized_face / 255.0
        reshaped_face = normalized_face.reshape(1, 48, 48, 1)

        # Emotion Prediction
        face_rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        resized_face = cv2.resize(face_rgb, (48, 48))
        normalized_face = resized_face / 255.0
        reshaped_face = normalized_face.reshape(1, 48, 48, 3)

        predictions = model.predict(reshaped_face, verbose=0)
        predicted_emotion = categories[np.argmax(predictions)]

        # Overlay emotion filter
        emotion_filter = filters.get(predicted_emotion, None)
        if emotion_filter is not None:
            overlay_filter(frame, emotion_filter, x, y, w, h)

        # Gender Prediction using HOG features
        gender_features = preprocess_for_gender_detection(face)
        predicted_gender = gender_model.predict(gender_features)[0]
        stabilized_gender = smooth_prediction(predicted_gender)
        gender_icon = male_icon if predicted_gender == 0 else female_icon

        # Overlay gender icon
        overlay_icon(frame, gender_icon, x, y, w, h)

    cv2.imshow('Real-Time Emotion and Gender Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
