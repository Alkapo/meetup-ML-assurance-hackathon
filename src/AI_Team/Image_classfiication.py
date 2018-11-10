# import the necessary packages


# CNN pcks section
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


# Extract and resize pcks section

import re
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
import numpy as np

import glob
print('loaded')

# Importer le modele
model = ResNet50(weights='imagenet')
model_for_output = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)

# Extract train dataset
file_list = glob.glob('../*.png')
