#!/usr/bin/env python3

import numpy as np
#from keras.models import Sequential
#from keras.layers import Dense, Conv2DTranspose, Conv2D, Flatten, MaxPooling2D, UpSampling2D, \
#       BatchNormalization, Activation,Dropout, regularizers, ConvLSTM2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75
set_session(tf.Session(config = config))

from utils import DataLoader
import build 

if __name__ == '__main__':
    model_name = 'convLSTM_external'
    feature_dir = 'feature/sequence_external/'

    train_size = 1605 * 5
    valid_size = 1820 * 5
    batch_size = 48
    epochs = 500

    train_loader = DataLoader(file_glob_pattern = '%s/*.train.*.npy' % feature_dir, batch_size = batch_size)
    valid_loader = DataLoader(file_glob_pattern = '%s/*.valid.*.npy' % feature_dir, batch_size = batch_size)
    model_ckpt = ModelCheckpoint('model/%s.h5' % model_name, verbose = 1, save_best_only = True)
    tensorboard = TensorBoard(log_dir='./logs/%s' % model_name , histogram_freq=0, write_graph=True, write_images=False)

    model = build.convLSTM_external()
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['accuracy'])
    model.fit_generator(train_loader, steps_per_epoch = train_size // batch_size, validation_data = valid_loader,\
                    validation_steps = valid_size // batch_size, epochs = epochs, callbacks = [model_ckpt, tensorboard])
