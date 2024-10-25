import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import tensorflow as tf
import tensorflow.keras.backend as K

IMG_SIZE = (48, 48)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2  

train_dir = 'C:/Users/user/emotion/archive/train/'
test_dir = 'C:/Users/user/emotion/archive/test/'

# Define a function to get labels from file paths
def get_label_from_filename(file_path):
    return os.path.basename(os.path.dirname(file_path))

# Retrieve and split the dataset
train_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(train_dir) for f in filenames]
test_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(test_dir) for f in filenames]
train_files, valid_files = train_test_split(train_files, test_size=VALIDATION_SPLIT, random_state=42)
print("Training data amount:", len(train_files))
print("Validation data amount:", len(valid_files))

# Extract labels
train_labels = [get_label_from_filename(f) for f in train_files]
valid_labels = [get_label_from_filename(f) for f in valid_files]
test_labels = [get_label_from_filename(f) for f in test_files]

# Map labels to integer indices
unique_labels = np.unique(train_labels)
label_to_index = {label: index for index, label in enumerate(unique_labels)}

# Convert labels to integers
train_labels = np.array([label_to_index[label] for label in train_labels])
valid_labels = np.array([label_to_index[label] for label in valid_labels])
test_labels = np.array([label_to_index[label] for label in test_labels])

# Calculate class weights based on the distribution of labels in the training set
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_dict = dict(enumerate(class_weights))

# Define class weights as an array matching the order of class indices
class_weights_array = np.array([class_weights_dict[i] for i in range(5)])

# Custom data generator
def custom_data_generator(image_files, labels, batch_size=32, target_size=(48, 48)):
    while True:
        for i in range(0, len(image_files), batch_size):
            batch_images = []
            batch_labels = labels[i:i + batch_size]
            
            for j in range(i, min(i + batch_size, len(image_files))):
                img = load_img(image_files[j], target_size=target_size)
                img = img_to_array(img)
                img = img / 255.0  # Normalize images
                batch_images.append(img)
            
            yield np.array(batch_images), np.array(batch_labels)

# Define custom weighted loss function
def custom_weighted_loss(weights):
    weights = K.constant(weights)
    
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, 'int32')
        sample_weights = K.gather(weights, y_true)
        scce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return scce * sample_weights
    
    return loss

# Define model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)

# Compile model with custom loss
model.compile(optimizer='adam',
              loss=custom_weighted_loss(class_weights_array),
              metrics=['accuracy'])

# Create generators
train_generator = custom_data_generator(train_files, train_labels, batch_size=BATCH_SIZE, target_size=IMG_SIZE)
valid_generator = custom_data_generator(valid_files, valid_labels, batch_size=BATCH_SIZE, target_size=IMG_SIZE)

# Train model
history = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=len(train_files) // BATCH_SIZE,
    validation_data=valid_generator,
    validation_steps=len(valid_files) // BATCH_SIZE
)

# Evaluate model on validation data
val_loss, val_accuracy = model.evaluate(valid_generator, steps=len(valid_files) // BATCH_SIZE)
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

#change to integer for class weights instead of one hot encoding 
# Validation loss: 1.3087188005447388
# Validation accuracy: 0.5139811038970947

# last layer to 512 from 128 
# Validation loss: 1.3513487577438354
# Validation accuracy: 0.529873251914978

# lower learning rate 0.0001 to 0.00005
# Validation loss: 1.4064137935638428
# Validation accuracy: 0.5369141101837158

# added batch normalization layers in CNN
# Validation loss: 1.9767787456512451
# Validation accuracy: 0.48259907960891724

# added even more conv2d layers in cnn 
# Validation loss: 2.241849899291992
# Validation accuracy: 0.572922945022583
