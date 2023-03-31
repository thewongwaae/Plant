import pathlib
import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.applications import EfficientNetV2M

# Download and prepare the dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

# Set image dimensions and batch size
batch_size = 32
height = 180
width = 180

# Create training and validation datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
	data_dir,
	validation_split=0.2,
	subset="training",
	seed=123,
	image_size=(height, width),
	batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
	data_dir,
	validation_split=0.2,
	subset="validation",
	seed=123,
	image_size=(height, width),
	batch_size=batch_size
)

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Define data augmentation layers
augment = Sequential([
	layers.RandomFlip("horizontal", input_shape=(height, width, 3)),
	layers.RandomRotation(0.1),
	layers.RandomZoom(0.1)
])

base_model = EfficientNetV2M(input_shape=(height, width, 3), include_top=False)

# Fine-tune the pre-trained model
model = Sequential([
	augment,
	layers.Rescaling(1./255),
	base_model,
	GlobalAveragePooling2D(),
	layers.Dense(128, activation='relu'),
	layers.Dense(len(train_ds.class_names), name="outputs")
])

base_model.trainable = False

# Compile the model
model.compile(
	optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
	# loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	loss=tf.keras.losses.BinaryCrossentropy(),
	metrics=["accuracy"]
)

# Train the model
epochs = 30
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Save the model
model.save("pretrained/")