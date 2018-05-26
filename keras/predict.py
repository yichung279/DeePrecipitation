#!/usr/bin/env python3
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report

if __name__ == '__main__':

    model = load_model('model/deconv.keras.h5')

    data = np.load('feature/valid.0.npy')
    x_te, y_te = np.split(data, [9], axis = 3)

    y_true = np.reshape(y_te, [-1])

    y_pred = model.predict(x_te)
    y_pred = np.argmax(y_pred, axis = 3)
    y_pred = np.reshape(y_pred, [-1])

    print(classification_report(y_true, y_pred, labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]))
