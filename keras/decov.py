#!/usr/bin/env python3

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2DTranspose, Conv2D, Flatten, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam

from utils import DataLoader

def build_model():
    model = Sequential()

    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu', input_shape = (72, 72, 9)))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))

    model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))

    model.add(Conv2DTranspose(filters = 512, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))

    model.add(Conv2DTranspose(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(Conv2DTranspose(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(UpSampling2D())

    model.add(Conv2DTranspose(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(Conv2DTranspose(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(UpSampling2D())

    model.add(Conv2DTranspose(filters = 128, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(Conv2DTranspose(filters = 128, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(UpSampling2D())

    model.add(Conv2DTranspose(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(Conv2DTranspose(filters = 15, kernel_size = (3, 3), padding = 'same', activation = 'softmax'))

    model.summary()

    return model

if __name__ == '__main__':

    train_loader = DataLoader(file_glob_pattern = 'feature/train.*.npy', batch_size = 100)
    valid_loader = DataLoader(file_glob_pattern = 'feature/valid.*.npy', batch_size = 200)

    model = build_model()
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['accuracy'])
    model.fit_generator(train_loader, steps_per_epoch = 10000 // 100, epochs = 100, validation_data = valid_loader, validation_steps = 11000 // 200 - 1, max_queue_size = 1)
