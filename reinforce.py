import os
import tkinter as tk
import shutil
from tkinter import filedialog
from PIL import Image, ImageTK

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