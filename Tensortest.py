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
	arr_boxes = boxes.numpy()[0].astype('int')
	arr_scores = scores.numpy()[0]

	img_boxes = rgb

	for score, (ymin,xmin,ymax,xmax), label in zip(arr_scores, arr_boxes, arr_labels):
		# if match score too low, skip
		if score < 0.65:
			continue
		
		# draw box and display classificaiton label
		score_txt = f'{100 * round(score,0)}'
		img_boxes = cv.rectangle(rgb,(xmin, ymax),(xmax, ymin),(0,255,0),1)      
		font = cv.FONT_HERSHEY_SIMPLEX
		cv.putText(img_boxes,label,(xmin, ymax-10), font, 0.5, (255,0,0), 1, cv.LINE_AA)
		cv.putText(img_boxes,score_txt,(xmax, ymax-10), font, 0.5, (255,0,0), 1, cv.LINE_AA)

	cv.imshow('Output', img_boxes)
	if cv.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv.destroyAllWindows()