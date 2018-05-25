#!/usr/bin/env python3

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2DTranspose, Conv2D, Flatten, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config = config))

from utils import DataLoader

def build_model():
    model = Sequential()

    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu', input_shape = (72, 72, 9)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(filters = 512, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D())

    model.add(Conv2DTranspose(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D())

    model.add(Conv2DTranspose(filters = 128, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(filters = 128, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D())

    model.add(Conv2DTranspose(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(filters = 15, kernel_size = (3, 3), padding = 'same', activation = 'softmax'))

    model.summary()

    return model

if __name__ == '__main__':

    batch_size = 96
    validation_size = 96
    epochs = 100

    train_loader = DataLoader(file_glob_pattern = 'feature/train.*.npy', batch_size = batch_size)
    valid_loader = DataLoader(file_glob_pattern = 'feature/valid.*.npy', batch_size = validation_size)
    model_ckpt = ModelCheckpoint('model/deconv.keras.h5', verbose = 1, save_best_only = True)

    model = build_model()
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['accuracy'])
    model.fit_generator(train_loader, steps_per_epoch = 10741 // batch_size, epochs = epochs, validation_data = valid_loader, validation_steps = 11482 // validation_size, callbacks = [model_ckpt])
