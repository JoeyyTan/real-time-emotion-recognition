import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib  # For saving the model

# Directories for male and female images
male_dir = "C:/Users/kevin/Documents/Gender Detection/archive/Training/male"
female_dir = "C:/Users/kevin/Documents/Gender Detection/archive/Training/female"

image_paths = []
labels = []

# Max number of images to load from each directory
max_images = 2000

# Load male images (limit to 2000)
print("Loading male images")
male_count = 0
for filename in os.listdir(male_dir):
    if male_count >= max_images:
        break
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_paths.append(os.path.join(male_dir, filename))
        labels.append(0)  # Male label = 0
        male_count += 1

# Load female images (limit to 2000)
print("Loading female images")
female_count = 0
for filename in os.listdir(female_dir):
    if female_count >= max_images:
        break
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_paths.append(os.path.join(female_dir, filename))
        labels.append(1)  # Female label = 1
        female_count += 1

labels = np.array(labels)

# Function to preprocess images
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (128, 128))
    return resized_image

# Preprocess all images
print("Pre-process")
processed_images = [preprocess_image(image_path) for image_path in image_paths]

# Function to extract HOG features
def extract_hog_features(image):
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

print("Extracting HOG features")
hog_features = [extract_hog_features(img) for img in processed_images]

# Convert HOG features to numpy array
hog_features = np.array(hog_features)

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)

# Train the SVM classifier
print("Training SVM")
svm = SVC(kernel='linear', C=1)
svm.fit(X_train, y_train)

# Evaluate the model
print("Evaluate Model")
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model to a file
print("Saving the model")
joblib.dump(svm, 'gender_classifier_model.pkl')  # Save as .pkl or .joblib

