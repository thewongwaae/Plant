import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from keras import layers
from keras.models import Sequential

# Path to tgz dataset, organise it in this format
# plants/
#   sawi/
#   lady_finger/
#   kangkung/
#   coriander/
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

# get number of images in the dataset
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# Eg displaying an image in the dataset
# roses = list(data_dir.glob('roses/*'))
# PIL.Image.open(str(roses[0]))

# For smaller datasets, batch size of 32 with 50 epochs is a good range
# For bigger ones, size 10 with 50 ~ 100 epochs should do
batch_size = 20
# Need standardized image dimensions for easier training
height = 180
width = 180

# vVlidation split of 0.2 means 80% for training and 20% for validation
training_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(height, width),
    batch_size=batch_size
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(height, width),
    batch_size=batch_size
)

# Check that the model has loaded the dataset correctly
class_names = training_dataset.class_names
print(class_names)

# Display 9 images from the training dataset
# plt.figure(figsize=(10, 10))
# for images, labels in training_dataset.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")

# Dataset.cache keeps the images in memory after they're loaded off disk during the first epoch. This will ensure the dataset does not become a bottleneck while training your model. If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache.
# Dataset.prefetch overlaps data preprocessing and model execution while training.
AUTOTUNE = tf.data.AUTOTUNE
training_dataset = training_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Normalise the data to reduce diversity
normalization_layer = layers.Rescaling(1./255)
normalized_ds = training_dataset.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
# Check pixel values
print(np.min(image_batch[0]), np.max(image_batch[0]))

# Prevent overfitting by changing up the training set a little
augment = Sequential([
    layers.RandomFlip("horizontal", input_shape=(height, width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# Building the model!
model = Sequential([
    augment,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), name="outputs")
])

# Set the optimiser for the model to "Adam" https://keras.io/api/optimizers/adam/
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
epochs = 10

history = model.fit(
  training_dataset,
  validation_data=validation_dataset,
  epochs=epochs
)

# Plot loss and accuracy on training and validation
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()


# Save the model
model.save("trained/")

# Reinforcement learning?