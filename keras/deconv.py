#!/usr/bin/env python3

import numpy as np
import argparse
from keras.models import Sequential
from keras.layers import Dense, Conv2DTranspose, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Dropout, Reshape
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config = config))

from utils import DataLoader

def build_model():
    model = Sequential()

    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', input_shape = (72, 72, 9)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = 'valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(filters = 512, kernel_size = (3, 3), padding = 'valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(filters = 256, kernel_size = (3, 3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(filters = 256, kernel_size = (3, 3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D())

    model.add(Conv2DTranspose(filters = 256, kernel_size = (3, 3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(filters = 256, kernel_size = (3, 3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D())

    model.add(Conv2DTranspose(filters = 128, kernel_size = (3, 3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(filters = 128, kernel_size = (3, 3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D())

    model.add(Conv2DTranspose(filters = 64, kernel_size = (3, 3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(filters = 3, kernel_size = (3, 3), padding = 'same', activation = 'softmax'))
    model.add(Reshape((-1, 3)))

    model.summary()

    return model

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('season', help='spring/summer/fall/winter/all')
    args = parser.parse_args()
    
    if args.season == 'spring':
        season = 'spring_'
        train_size = 5388
        valid_size = 5703 
    elif args.season == 'summer':
        season = 'summer_'
        train_size = 1964
        valid_size = 2202
    elif args.season == 'fall':
        season = 'fall_'
        train_size = 764
        valid_size = 725
    elif args.season == 'winter':
        season = 'winter_'
        train_size = 5411
        valid_size = 5827
    elif args.season == 'all':
        season = ''
        train_size = 13527 
        valid_size = 14457
    else:
        os._exit(0)
    model_name = 'd_0.2_w_1_2_10'

    batch_size = 72
    epochs = 100

    class_weight = np.zeros((72 * 72, 3))
    class_weight[:, 0] += 1
    class_weight[:, 1] += 2
    class_weight[:, 2] += 10

    train_loader = DataLoader(file_glob_pattern = 'feature/' + season + 'train.*.npy', batch_size = batch_size)
    valid_loader = DataLoader(file_glob_pattern = 'feature/' + season + 'valid.*.npy', batch_size = batch_size)
    model_ckpt = ModelCheckpoint('model/'+ model_name + '.' + season + 'model.keras.h5', verbose = 1, save_best_only = True)
    tensorboard = TensorBoard(log_dir='./logs/' + season + model_name, histogram_freq=0, write_graph=True, write_images=False)

    model = build_model()
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['accuracy'])
    model.fit_generator(train_loader, class_weight=class_weight, steps_per_epoch = train_size // batch_size, epochs = epochs, validation_data = valid_loader, validation_steps = valid_size // batch_size
           , callbacks = [model_ckpt, tensorboard])
