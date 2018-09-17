#!/usr/bin/env python3

import numpy as np
import argparse
from keras.models import Sequential
from keras.layers import Dense, Conv2DTranspose, Conv2D, Flatten, MaxPooling2D, UpSampling2D, \
        BatchNormalization, Activation,Dropout, regularizers, ConvLSTM2D
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

    model.add(ConvLSTM2D(
        filters = 64, 
        kernel_size = (3, 3),
        padding = 'same',  
        input_shape = (3, 72, 72, 3),    # channel_last as defult
        return_sequences = True,
        stateful = False
    ))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3), padding = 'same', return_sequences = True, stateful = False))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3), padding = 'same', return_sequences = False, stateful = False))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters = 3, kernel_size = (3, 3), padding = 'same', activation = 'softmax'))

    model.summary()

    return model

if __name__ == '__main__':
    
    '''
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
    '''
    model_name = 'convLSTM2D'

    train_size = 1054 * 5
    valid_size = 1186 * 5
    batch_size = 32
    epochs = 100

    train_loader = DataLoader(file_glob_pattern = 'no_compensate_feature/*.train.*.npy', batch_size = batch_size)
    valid_loader = DataLoader(file_glob_pattern = 'no_compensate_feature/*.valid.*.npy', batch_size = batch_size)
    # model_ckpt = ModelCheckpoint('model/no_compensate.model.keras.h5', verbose = 1, save_best_only = True)
    tensorboard = TensorBoard(log_dir='./logs/no_convLSTM_compensate', histogram_freq=0, write_graph=True, write_images=False)

    model = build_model()
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['accuracy'])
    model.fit_generator(train_loader, steps_per_epoch = train_size // batch_size, epochs = epochs, validation_data = valid_loader, validation_steps = valid_size // batch_size)
    #        , callbacks = [model_ckpt, tensorboard])
