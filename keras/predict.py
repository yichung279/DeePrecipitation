#!/usr/bin/env python3
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report
from math import sqrt

from utils import label2pixel, write_image
import os


def load():
    feature_dir = 'feature/'

    img = []
    y_te = []
    for i in 'ACE':
        datas = np.load('%s/%s.valid.0.npy' % (feature_dir, i))
        for data in datas:
            img.append(data.get('x'))
            y_te.append(data.get('label'))
    
    x_te = np.array(img)
    y_te = np.array(y_te)
    
    return x_te, y_te
    
def compile_statistics(model_name, x_te, y_te):
    model = load_model('model/%s.h5' % model_name)

    y_true = np.reshape(y_te, [-1])

    y_pred = model.predict(x_te)
    y_pred = np.argmax(y_pred, axis = 3)
    y_pred = np.reshape(y_pred, [-1])

    print(classification_report(y_true, y_pred, labels = [0,1,2]))

def draw(model_name, x_te, y_te):
    model = load_model('model/%s.h5' % model_name)
    output_dir = 'output/' + model_name

    imgs_pred = model.predict(x_te[:50])
    imgs_pred = np.argmax(imgs_pred, axis = 3)

    y_te = np.reshape(y_te, [-1, 72, 72])
    print(y_te.shape)
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for idx, img_pred in enumerate(imgs_pred):
        write_image('%s/valid.%03d.png' % (output_dir, idx), img_pred)
        write_image('%s/true.%03d.png' % (output_dir, idx), y_te[idx])

if __name__ == '__main__':
    x_te, y_te = load()
    model_name = 'convLSTM'
    compile_statistics(model_name, x_te, y_te)
    draw(model_name, x_te, y_te)

