import h5py, os
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.vgg16 import VGG16

model = VGG16(weights='imagenet', include_top=True)

with h5py.File('output/preprocessed.h5', 'r') as h5_file:
    ttt = h5_file.keys()
    dset = h5_file['images']
    x =dset[2]

x = np.expand_dims(x, axis=0)
predictions = model.predict(x)

textLabels = decode_predictions(predictions)

print('features:', predictions, np.argmax(predictions), textLabels)