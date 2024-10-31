import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback

# Define parameters
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2  

# Directories
train_dir = 'C:/Users/Melvin Tang/OneDrive/Codes/I2ML/FINALPRO/archive/train'
test_dir = 'C:/Users/Melvin Tang/OneDrive/Codes/I2ML/FINALPRO/archive/test'

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.20,
    zoom_range=0.20,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT  
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',  
    shuffle=False  # Shuffle set to False for validation
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Categories should be defined based on your class labels
categories = list(train_generator.class_indices.keys())

# Class Weights Calculation
num_train_angry = 1000
num_train_fear = 1000
num_train_happy = 1000
num_train_neutral = 1000
num_train_sad = 1000

class_indices = train_generator.class_indices
print(f"Class Indices: {class_indices}")

num_train_samples = {
    'angry': num_train_angry,
    'fear': num_train_fear,
    'happy': num_train_happy,
    'neutral': num_train_neutral,
    'sad': num_train_sad
}

labels = []
for class_name in class_indices:
    labels += [class_indices[class_name]] * num_train_samples[class_name]

classes = np.unique(labels)
class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=labels)
class_weights_dict = dict(zip(classes, class_weights))
print(f"Class Weights: {class_weights_dict}")

class_weights_array = np.array([class_weights_dict[class_indices[class_name]] for class_name in sorted(class_indices, key=class_indices.get)])

def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)
    
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weighted_ce = cross_entropy * weights
        loss = K.sum(weighted_ce, axis=-1)
        return loss
    
    return loss

from tensorflow.keras import regularizers

l2_strength = 0.01  # Adjust this value as needed

# Model Definition
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3),
                  kernel_regularizer=regularizers.l2(l2_strength)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu',
                  kernel_regularizer=regularizers.l2(l2_strength)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu',
                  kernel_regularizer=regularizers.l2(l2_strength)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu',
                  kernel_regularizer=regularizers.l2(l2_strength)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    
    layers.Dense(512, activation='relu',
                 kernel_regularizer=regularizers.l2(l2_strength)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(5, activation='softmax',
                 kernel_regularizer=regularizers.l2(l2_strength))
])

# Set a higher initial learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.000075)

# Compile the model
model.compile(optimizer=optimizer,
              loss=weighted_categorical_crossentropy(class_weights_array),
              metrics=['accuracy'])

# Learning Rate Scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.45, patience=5, min_lr=1e-6)

# Custom callback to display learning rate
class LearningRateLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate.numpy()
        print(f"Learning Rate at the end of epoch {epoch + 1}: {lr:.6f}")

# Train the model
history = model.fit(
    train_generator,
    epochs= 200,
    validation_data=validation_generator,
    callbacks=[lr_scheduler, LearningRateLogger()]  # Add the learning rate logger
)

val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation loss: {val_loss}")
print(f"Validation accuracy: {val_accuracy}")

predictions = model.predict(validation_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_generator.classes
class_labels = {v: k for k, v in train_generator.class_indices.items()}

# for i in range(len(predicted_classes)):
#     print(f"Predicted: {class_labels[predicted_classes[i]]}, True: {class_labels[true_classes[i]]}")

# # Visualize misclassifications
# for i in range(len(predicted_classes)):
#     if predicted_classes[i] != true_classes[i]:
#         img, _ = validation_generator[i]
#         plt.imshow(img[0])
#         plt.title(f"True: {class_labels[true_classes[i]]}, Predicted: {class_labels[predicted_classes[i]]}")
#         plt.show()

def plot_accuracy(history):
    """
    Plots training and validation accuracy.
    """
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss(history):
    """
    Plots training and validation loss.
    """
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_test_set(model, test_generator, steps, categories):
    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_generator, steps=steps)
    print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')
    
    # Make predictions on the test dataset
    predictions = model.predict(test_generator, steps=steps)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels from test generator
    true_classes = test_generator.classes
    
    # Generate classification report
    print(classification_report(true_classes, predicted_classes, target_names=categories))
    
    # Create a confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)

    # Plot the confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Test Set)')
    plt.show()


# Plot accuracy
plot_accuracy(history)

# Plot loss
plot_loss(history)

#test_image_count = sum([len(os.listdir(d)) for d in test_dir])
#steps = np.ceil(test_image_count / BATCH_SIZE).astype(int)

test_image_count = sum([len(files) for r, d, files in os.walk(test_dir)])  # Corrected file count

# Ensure steps is an integer
steps = int(np.ceil(test_image_count / BATCH_SIZE))  # Correct calculation of steps

evaluate_test_set(model, test_generator, steps, categories)