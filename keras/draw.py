import numpy as np
import os

from keras.models import load_model
from utils import label2pixel, write_image


if __name__ == '__main__':

    if not os.path.isdir('output'):
        os.makedirs('output')

    data = np.load('feature/valid.0.npy')
    x_te, y_te = np.split(data, [9], axis = 3)
    y_te = np.reshape(y_te, [-1, 72, 72])

    model = load_model('model/deconv.keras.h5')
    pred = model.predict(x_te[:50])
    pred = np.argmax(pred, axis = 3)

    for idx, img_pred in enumerate(pred):
        write_image('output/valid.%03d.png' % idx, img_pred)
        write_image('output/true.%03d.png' % idx, y_te[idx])
