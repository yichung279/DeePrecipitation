
from keras.models import Sequential
from keras.layers import Dense, Conv2DTranspose, Conv2D, Flatten, MaxPooling2D, UpSampling2D, \
       BatchNormalization, Activation,Dropout, regularizers, ConvLSTM2D

def convLSTM():
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

