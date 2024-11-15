from tensorflow.keras.models import load_model

# Load your model with custom objects if necessary
model = load_model('C:/Users/LENOVO/Downloads/real-time-emotion-recognition/cnn_trial.h5', custom_objects={'weighted_categorical_crossentropy': weighted_categorical_crossentropy})

# Save as .h5 file
model.save('C:/Users/LENOVO/Downloads/real-time-emotion-recognition/cnn_trial.h5')

# Or save as .keras
model.save('C:/Users/LENOVO/Downloads/real-time-emotion-recognition/cnn_trial.keras')

# Or save as SavedModel format
#model.save('C:/Users/LENOVO/Downloads/real-time-emotion-recognition/cnn_trial', save_format='tf')
