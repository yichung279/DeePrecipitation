#!/usr/bin/env python3
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report

if __name__ == '__main__':

    # model = load_model('model/dropout0.2.model.keras.h5')
    model = load_model('model/compensate.deconv.model.keras.h5')

    data = []
    for i in 'ACE':
        data.extend(np.load('feature/%s.valid.0.npy' % i))
    x_te, y_te = np.split(data, [9], axis = 3)
    
    y_true = np.reshape(y_te, [-1])
    ''' 

    img1, img2, img3 = np.split(imgs, 3, axis = 3)

    x_te = [0] * 3000
    for idx in range(3000):
        x_te[idx] = [img1[idx], img2[idx], img3[idx]]
    '''
    y_pred = model.predict(np.array(x_te))
    y_pred = np.argmax(y_pred, axis = 3)
    y_pred = np.reshape(y_pred, [-1])

    print(classification_report(y_true, y_pred, labels = [0,1,2]))
