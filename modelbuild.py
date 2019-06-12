'''
Build a model
'''
from tensorflow.python.keras.applications import DenseNet121
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.python.keras import regularizers

def build_model(imagesize, n_classes):
    model = Sequential()
    # We use pretrained densenet as our base
    model.add(DenseNet121(input_shape=(imagesize, imagesize, 3),
                                       weights = 'imagenet',
                                       include_top=False,
                                       pooling='avg'))
    model.add(Flatten())

    # Freeze the densenet, since we wanna use their weights,
    # we only train the new layers we are adding down below
    model.layers[0].trainable = False
    model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax', kernel_initializer='random_uniform', bias_initializer='random_uniform', bias_regularizer=regularizers.l2(0.01), name='Output'))

    return model
