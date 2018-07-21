#!/usr/bin/env python3
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report

if __name__ == '__main__':

    model = load_model('model/d_0.2_w_1_1_100.model.keras.h5')
    
    season=''
    num=5

    data = []
    for i in range(num):
        data.extend(np.load('feature/'+season+'valid.'+str(i*3)+'.npy'))
    x_te, y_te = np.split(data, [9], axis = 3)

    y_te = y_te.reshape((-1, 72*72))

    y_true = np.reshape(y_te, [-1])

    y_pred = model.predict(x_te)
    y_pred = np.argmax(y_pred, axis = 2)
    y_pred = np.reshape(y_pred, [-1])

    print(classification_report(y_true, y_pred, labels = [0,1,2]))
