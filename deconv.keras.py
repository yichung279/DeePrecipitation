import random
import numpy as np
from glob import glob
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2DTranspose, Conv2D,Flatten, MaxPooling2D, UpSampling2D
from keras.utils import  np_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

global filename
global data
global batch_count
filename = 'data/dataEven_1.npy'
data = np.load(filename)
batch_count = 0
#def load_data():
def arr1Dto2D(arr):
    arr2D = [[None] *288] *288
    for i in range(288):
        for j in range(288):
            arr2D[i][j] = arr[288*i + j]
    return arr2D

def batch_generator(batch_siize, train='True'):

    if 1000 % batch_size != 0:
        print('wrong batch size!')
        os._exit()
    if train :
        global data
        global filename
        global batch_count
        batch_count += 1

        file_size = int(1000 / batch_size) # how much batches in a file
        file_num = int(batch_count / file_size) % 6
        batch_num = int(batch_count % file_size)

        filenames = glob('data/dataEven_*.npy')
        filenames.sort()
        # try to do few times of load
        if filenames[file_num] != filename:
            filename = filenames[file_num]

            data = np.load(filename)
            np.random.shuffle(data)
    else :
        batch_num = 0

        filenames = glob('data/dataOdd_*.npy')
        filenames.sort()
        file_num = np.random.choice(6)
        filename = filenames[file_num]
        data = np.load(filename)
        np.random.shuffle(data)

    label = []
    x = []
    label_holder = [0] * 15

    for i in range(batch_num * batch_size, (batch_num+1) * batch_size):
        #label
        sublabel = []
        for j in range(data.shape[2]):
            label_holder[data[i][1][j]] = 1
            sublabel.append(label_holder)
            label_holder = [0] * 15
        label.append(arr1Dto2D(sublabel))
        #input
        subX = []
        chanel1 = arr1Dto2D(data[i][1])
        chanel2 = arr1Dto2D(data[i][2])
        chanel3 = arr1Dto2D(data[i][3])
        subX.append(chanel1)
        subX.append(chanel2)
        subX.append(chanel3)
        x.append(np.transpose(subX))

    label = np.array(label)
    x = np.array(x)
    return [x, label]
 

def build_model():
    #建立模型
    model = Sequential()
    #將模型疊起
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(288, 288, 3), activation='relu')) 
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=4096, kernel_size=(9, 9), padding='valid', activation='relu'))
    model.add(Conv2D(filters=4096, kernel_size=(1, 1), padding='valid', activation='relu'))

    model.add(Conv2DTranspose(filters=512, kernel_size=(9, 9), padding='valid', activation='relu')) 

    model.add(UpSampling2D())    

    model.add(Conv2DTranspose(filters=512, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2DTranspose(filters=512, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2DTranspose(filters=512, kernel_size=(3, 3), padding='same', activation='relu')) 
    
    model.add(UpSampling2D())    

    model.add(Conv2DTranspose(filters=512, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2DTranspose(filters=512, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2DTranspose(filters=512, kernel_size=(3, 3), padding='same', activation='relu')) 

    model.add(UpSampling2D())
 
    model.add(Conv2DTranspose(filters=256, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2DTranspose(filters=256, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2DTranspose(filters=256, kernel_size=(3, 3), padding='same', activation='relu')) 

    model.add(UpSampling2D())

    model.add(Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same', activation='relu')) 
 
    model.add(UpSampling2D())

    model.add(Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='relu')) 
    model.add(Conv2DTranspose(filters=15, kernel_size=(3, 3), padding='same', activation='softmax')) 

    model.summary()

    return model
if __name__ == '__main__':
    batch_size = 20
    epochs = 100
    file_nums = [0, 1, 2, 3, 4, 5] 
    #model = build_model()
    model = load_model('model/latest_model.h5')
    #開始訓練模型
    model.compile(loss='categorical_crossentropy',optimizer="adam", metrics=['categorical_accuracy'])
    for e in range(epochs):
        for i in range(int(6000 / batch_size)): 
            #! Todo: step by step (for memery)
            x_train, y_train = batch_generator(batch_size)
            print("epoch %d,step %d:" % (e,i))
            train = model.train_on_batch(x_train, y_train)
            print('train:', train)
            x_test, y_test = batch_generator(batch_size, train=False)
            valid = model.evaluate(x_test, y_test)
            print('valid:' ,valid)
            if i == 0:
                model.save('model/first_model.h5')
                print('model saved')
            elif i % 50 == 0 :
                model.save('model/latest_model.h5')
                print('model saved')
