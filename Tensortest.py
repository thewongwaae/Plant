import tensorflow_hub as hub
import tensorflow as tf
import cv2 as cv
import numpy
import pandas as pd

# force python env to use first GPU if available
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# check if Tensorflow detects GPU
if len(tf.config.list_physical_devices('GPU')) > 0:
	cv.useOptimized()

# detector follows a dataset identical to labels.csv
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
labels = pd.read_csv('labels.csv', sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']

cap = cv.VideoCapture(0)

width = 512
height = 512

while(True):
	# capture frame by frame
	ret, frame = cap.read()
	# dynamic resize depending on camera resolution
	resized = cv.resize(frame, (width, height))
	# convert to RGB
	# will need to convert to something else (HSV??) for plant
	rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)

	# float conversion and convert img to tensor image for faster processing
	rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
	# add *dims* to rgb_tensor
	rgb_tensor = tf.expand_dims(rgb_tensor, 0)

	boxes, scores, classes, detects = detector.__call__(rgb_tensor)
	#boxes, scores, classes, detects = detector(rgb_tensor, training=False)

	filter = [1, 64] # person and potted plant

	arr_labels = classes.numpy().astype('int')[0]

	filtered_indices = [i for i, label_id in enumerate(arr_labels) if label_id in filter]
	# issue here
	filtered_labels = [labels[i - 1] for i in filtered_indices]
	arr_boxes = boxes.numpy()[0].astype('int')
	filtered_boxes = arr_boxes[filtered_indices]
	arr_scores = scores.numpy()[0]
	filtered_scores = arr_scores[filtered_indices]

	img_boxes = rgb

	for score, (ymin, xmin, ymax, xmax), label in zip(filtered_scores, filtered_boxes, filtered_labels):
		# adjust for how similar an item has to be for it to be categorised
		if score < 0.5:
			continue
			
		# display score as percentage
		score_txt = f'{100 * round(score, 0)}'
		img_boxes = cv.rectangle(rgb, (xmin, ymax), (xmax, ymin), (0, 255, 0), 1)
		font = cv.FONT_HERSHEY_SIMPLEX
		cv.putText(img_boxes, label, (xmin, ymax-10), font, 0.5, (255, 0, 0), 1, cv.LINE_AA)
		cv.putText(img_boxes, label, (xmax, ymax-10), font, 0.5, (255, 0, 0), 1, cv.LINE_AA)

	cv.imshow('Output', img_boxes)
	if cv.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv.destroyAllWindows()