import training_win
import numpy as np
import tensorflow as tf
import cv2 as cv

i = 0

model = tf.keras.models.load_model("trained/")

# Normalize the image input
def preprocess(image, height, width):
	resized = cv.resize(image, (width, height))
	normalized = resized / 255.0
	input_data = np.expand_dims(normalized, axis=0)
	return input_data

# Use trained model to classify the image based on trained
def process(image, model, height, width, threshold):
	input_data = preprocess(image, height, width)
	prediction = model.predict(input_data)
	if prediction[0][0] >= threshold:
		cv.imwrite("predictions/file" + i + ".jpg", image)



height = 256
width = 256
threshold = 0.8

cap = cv.VideoCapture(0)

while True:
	ret, frame = cap.read()
	if not ret:
		break
	process(frame, model, height, width, threshold)
	cv.imshow("Live Video Feed", frame)
	if cv.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv.destroyAllWindows()