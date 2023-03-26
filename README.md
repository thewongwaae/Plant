# Plant Health Monitor

## The Idea
Looking to automate observation of plants under the care of busy humans such as us programmers, thus we must spend hours working on something that helps us complicate gardening ourselves.

## The Plan
- Raspberry Pi equipped with a camera
- Object detection (Plant classification) Tensorflow model on the setup
- Get plant health updates through an app

## The Theory
There seems to be no plant-specific model I can just grab from the internet ( rather I haven't looked into it ), therefore a model shall be trained on specific plants Mars will be planting. The object detection model will be trained to recognise the different plants under any light circumstance. <br />

Once the models are trained and the setup is ready to use, the setup will take still photos of the plants ( every 15~30 minutes ) and throw it over to the model. This will give updates to the app through an API. <br />

Additional features ( to be added? )
- [ ] App: Live feed of plants
- [ ] Setup: Automated watering

## Setup
TLDR
```
pip install tensorflow tensorflow-hub matplotlib numpy Pillow keras pandas
```

OpenCV currently `19/3/2023` only supports up until Python 3.9, so it is recommended to have a Python 3.9 installation for this to run properly.

Tensorflow will make use of your GPU if it is compatible with the correct packages installed. Read more [here](https://www.tensorflow.org/install/pip).

The program needs these packages to run 
```python
import tensorflow_hub as hub
import tensorflow as tf
import cv2 as cv
import numpy
import pandas as pd
```

### Windows
Tensorflow and Tensorflow_hub
```
pip install tensorflow
pip install tensorflow_hub
```

OpenCV [installation guide](https://docs.opencv.org/4.x/d5/de5/tutorial_py_setup_in_windows.html)

Numpy
```
pip install numpy
```

Pandas
```
pip install pandas
```
