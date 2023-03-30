import os
import tkinter as tk
import shutil
import tensorflow as tf
from keras import layers
from tkinter import filedialog
from PIL import Image, ImageTk

height = 256
width = 256
batch_size = 32

def load_next_img(label, paths, index):
	image = Image.open(paths[index])
	image.thumbnail((500, 500))
	photo = ImageTk.PhotoImage(image)
	label.config(image=photo)
	label.image = photo

def handle_feedback(label, paths, index, feedback):
	print(f"Image: {paths[index]}, Feedback: {feedback}")
	if feedback == True:
		shutil.move(paths[index], "reinforced/right/")
	else:
		shutil.move(paths[index], "reinforced/wrong/")

	if index < len(paths) - 1:
		load_next_img(label, paths, index + 1)
		return index + 1
	else:
		root.quit()

def on_yes():
	global index
	index = handle_feedback(label, paths, index, True)

def on_no():
	global index
	index = handle_feedback(label, paths, index, False)

def finetune(model_path):
	global index
	global label
	global paths

	# Display manual verification window
	root = tk.Tk()
	root.title("Image verification")

	label = tk.Label(root)
	label.pack()

	dir = "images"
	paths = [os.path.join(dir, f) for f in os.listdir(dir)]

	index = 0
	load_next_img(label, paths, index)

	b_right = tk.Button(root, text="Yes", command=on_yes)
	b_wrong = tk.Button(root, text="No", command=on_no)
	b_right.pack(side=tk.LEFT, padx=10)
	b_wrong.pack(side=tk.RIGHT, padx=10)

	root.mainloop()

	# Reinforce the model
	model = tf.keras.models.load_model(model_path)

	new_train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
		directory=["reinforced/right/", "reinforced/wrong/"],
		labels="inferred",
		label_mode="binary",
		image_size=(height, width),
		batch_size=batch_size
	)

	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
		loss=tf.keras.losses.BinaryCrossentropy(),
		metrics=["accuracy"]
	)
	
	model.fit(new_train_dataset, epochs=10)

	# Save the fine-tuned model
	model.save(model_path)

finetune("trained/")