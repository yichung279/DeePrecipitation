import numpy as np
from glob import glob
from keras.models import load_model
import os

import trans

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    model = load_model('model/latest_model.h5')
    data = np.load('data/dataEven_1.npy')[0]
    images = model.predict(trans.preProcess(data))
    print(images)
    for i, image in enumerate(images):
        trans.postProcess(image, 'image%d.png' %(i+1))
        print('img saved')
