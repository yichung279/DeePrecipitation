import numpy as np
import os

from keras.models import load_model
from utils import label2pixel, write_image


if __name__ == '__main__':

    if not os.path.isdir('output'):
        os.makedirs('output')

    data = np.load('feature/valid.0.npy')
    x_te, _ = np.split(data, [9], axis = 3)

    model = load_model('model/deconv.keras.h5')
    pred = model.predict(x_te[:50])

    for idx, img_pred in enumerate(pred):
        write_image('output/valid.%03d.png' % idx, img_pred)
