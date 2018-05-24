import random
import numpy as np
from glob import glob
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2DTranspose, Conv2D,Flatten, MaxPooling2D, UpSampling2D
from keras.utils import  np_utils
from mlxtend.preprocessing import shuffle_arrays_unison

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

global batch_count
batch_count = 0

def batch_generator(file_siize):
    if 1000 % file_size != 0:
        print('wrong size!')
        os._exit()

    global batch_count
    file_num = int(batch_count % 11)
    batch_count += 1

    filenames = glob('data/radarTrend.train.*.input.npy') 
    filenames.sort()
    x_train = np.load(filenames[file_num])
    
    filenames = glob('data/radarTrend.train.*.label.npy') 
    filenames.sort()
    y_train = np.load(filenames[file_num])

    x_train, y_train = shuffle_arrays_unison(arrays=[x_train, y_train])
    x_train = np.split(x_train, file_size)
    y_train = np.split(y_train, file_size)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    file_num = np.random.choice(10)

    filenames = glob('data/radarTrend.test.*.input.npy')
    x_valid = np.load(filenames[file_num])

    filenames = glob('data/radarTrend.test.*label.npy')
    y_valid = np.load(filenames[file_num])

    x_valid, y_valid = shuffle_arrays_unison(arrays=[x_valid, y_valid])
    x_valid = np.split(x_valid, file_size)
    y_valid = np.split(y_valid, file_size)
    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)

    return x_train, y_train, x_valid, y_valid

def build_model():
    #建立模型
    model = Sequential()
    #將模型疊起
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(144, 144, 3), activation='relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')) 
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=4096, kernel_size=(9, 9), padding='valid', activation='relu'))
    #model.add(Conv2D(filters=4096, kernel_size=(1, 1), padding='valid', activation='relu'))

    model.add(Conv2DTranspose(filters=512, kernel_size=(9, 9), padding='valid', activation='relu')) 
    model.add(UpSampling2D())    

    model.add(Conv2DTranspose(filters=512, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2DTranspose(filters=512, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(UpSampling2D())    

    #model.add(Conv2DTranspose(filters=512, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2DTranspose(filters=512, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(UpSampling2D())
 
    model.add(Conv2DTranspose(filters=256, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2DTranspose(filters=256, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(UpSampling2D())

    #model.add(Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same', activation='relu')) 
    #model.add(UpSampling2D())

    model.add(Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2DTranspose(filters=15, kernel_size=(3, 3), padding='same', activation='softmax')) 

    model.summary()

    return model
if __name__ == '__main__':
    batch_size = 50
    file_size = int(1000/batch_size)
    epochs = 100
    model = build_model()
    #開始訓練模型
    model.compile(loss='categorical_crossentropy',optimizer="adam", metrics=['categorical_accuracy'])
    for e in range(epochs):
        for i in range(11):
            x_train, y_train, x_valid, y_valid = batch_generator(file_size) 
            for j in range(file_size): 
                train = model.train_on_batch(x_train[j], y_train[j])
                #print('epoch :%d,step, %d,  train: %f valid %f'%(e, 1000*i+j , train[0]))
                print(train[0])
                valid = model.evaluate(x_valid[j], y_valid[j])
                print('              +----> valid: %f'%(valid[0]))
                if i * (file_size) + j == 0:
                    model.save('model/first_model.h5')
                    print('model saved')
                elif i * (file_size) + j % 50 == 0 :
                    model.save('model/latest_model.h5')
                    print('model saved')
