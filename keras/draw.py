import numpy as np
import os

from keras.models import load_model
from utils import label2pixel, write_image


if __name__ == '__main__':
    model_name = 'no_compensate.deconv' 
    output_dir = 'deconv.no_compensate.output'
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    data = np.load('feature/B.valid.0.npy')
    x_te, y_te = np.split(data, [9], axis = 3)
    ''' 
    imgs = [0] * x_te.shape[0]                                                                                                                           
    for idx in range(x_te.shape[0]):
            imgs[idx] = [img1[idx], img2[idx], img3[idx]]
    
    imgs = np.array(imgs)
    '''
    y_te = np.reshape(y_te, [-1, 72, 72])
    model = load_model('model/' + model_name + '.model.keras.h5')
    pred = model.predict(x_te[:50])
    pred = np.argmax(pred, axis = 3)

    for idx, img_pred in enumerate(pred):
        write_image('%s/valid.%03d.png' % (output_dir, idx), img_pred)
        write_image('%s/true.%03d.png' % (output_dir, idx), y_te[idx])
