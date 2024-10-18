import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K

IMG_SIZE = (48, 48)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2  

train_dir = 'C:/Users/user/emotion/archive/train/'
test_dir = 'C:/Users/user/emotion/archive/test/'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
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
        # Clip predictions to prevent NaNs and Infs
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Calculate cross-entropy
        cross_entropy = -y_true * K.log(y_pred)
        
        # Apply the weights
        weighted_ce = cross_entropy * weights
        
        # Sum over classes
        loss = K.sum(weighted_ce, axis=-1)
        return loss
    
    return loss

from tensorflow.keras import models, layers

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # layers.Dropout(0.5),  # Add dropout layer after the dense layer
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(5, activation='softmax') 
])


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer='adam',
              loss=weighted_categorical_crossentropy(class_weights_array),
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator,
)

val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation loss: {val_loss}")
print(f"Validation accuracy: {val_accuracy}")

# predictions = model.predict(validation_generator)
# predicted_classes = np.argmax(predictions, axis=1)
# true_classes = validation_generator.classes
# class_labels = {v: k for k, v in train_generator.class_indices.items()}

# for i in range(len(predicted_classes)):
#     print(f"Predicted: {class_labels[predicted_classes[i]]}, True: {class_labels[true_classes[i]]}")


# # Visualize misclassifications
# for i in range(len(predicted_classes)):
#     if predicted_classes[i] != true_classes[i]:
#         # Plot the image
#         img, _ = validation_generator[i]
#         plt.imshow(img[0])  # Display the image
#         plt.title(f"True: {class_labels[true_classes[i]]}, Predicted: {class_labels[predicted_classes[i]]}")
#         plt.show()


# original 
# Validation loss: 1.366945743560791
# Validation accuracy: 0.44612929224967957

# add optimizer 
# Validation loss: 1.3994815349578857
# Validation accuracy: 0.43535515666007996

#changed num of samples to 1000 each with optimizer 
# Validation loss: 1.2316433191299438
# Validation accuracy: 0.5027933120727539

# dropout 0.25
# Validation loss: 1.292473316192627
# Validation accuracy: 0.46737179160118103

# no dropout, epoch 25
# Validation loss: 1.2633336782455444
# Validation accuracy: 0.4825384020805359

# dropout 0.5, epoch 25
# Validation loss: 1.3869889974594116
# Validation accuracy: 0.4148872494697571