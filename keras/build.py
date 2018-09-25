
from keras.models import Sequential
from keras.layers import Dense, Conv2DTranspose, Conv2D, BatchNormalization, Activation,Dropout, ConvLSTM2D

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

def convLSTM_external():
    model = Sequential()

    model.add(ConvLSTM2D(
        filters = 64, 
        kernel_size = (5, 5),
        padding = 'valid',  
        input_shape = (3, 80, 80, 3),    # channel_last as defult
        return_sequences = True,
        stateful = False
    ))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3), padding = 'valid', return_sequences = True, stateful = False))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3), padding = 'valid', return_sequences = False, stateful = False))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters = 3, kernel_size = (3, 3), padding = 'same', activation = 'softmax'))

    model.summary()

    return model

def convLSTM_deconv():
    model = Sequential()

    model.add(ConvLSTM2D(
        filters = 64, 
        kernel_size = (5, 5),
        padding = 'valid',  
        input_shape = (3, 72, 72, 3),    # channel_last as defult
        return_sequences = True,
        stateful = False
    ))
    model.add(BatchNormalization())
        
    model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3), padding = 'valid', return_sequences = True, stateful = False))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3), padding = 'valid', return_sequences = False, stateful = False))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(filters = 32, kernel_size = (3, 3), padding = 'valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
   
    model.add(Conv2DTranspose(filters = 32, kernel_size = (3, 3), padding = 'valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
   
    model.add(Conv2DTranspose(filters = 64, kernel_size = (5, 5), padding = 'valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters = 3, kernel_size = (3, 3), padding = 'same', activation = 'softmax'))
    
    model.summary()

    return model
