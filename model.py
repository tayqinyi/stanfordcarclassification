'''
Build a model
'''
from tensorflow.python.keras.applications import densenet
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.python.keras import regularizers

def build_model(imagesize, n_classes):
    model = Sequential()
    model.add(densenet.DenseNet121(input_shape=(imagesize, imagesize, 3),
                                       weights = 'imagenet',
                                       include_top=False,
                                       pooling='avg'))
    model.add(Flatten())

    # Freeze the densenet, since we wanna use their weights,
    # we only train the new layers we are adding down below
    model.layers[0].trainable = False
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax', name='Output'))

    return model
