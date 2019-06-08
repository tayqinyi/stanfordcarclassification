'''
BUild a model
'''
from tensorflow.python.keras.applications import densenet
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout

def build_model(imagesize, n_classes):
    model = Sequential()
    model.add(densenet.DenseNet121(input_shape=(imagesize, imagesize, 3),
                                   include_top=False,
                                   pooling='avg'))

    # Freeze the densenet, since we wanna use their weights,
    # we only train the new layers we are adding down below
    model.layers[0].trainable = False
    model.add(Dense(256, activation='relu', name='Dense_Intermediate'))
    model.add(Dropout(0.1, name='Dropout_Regularization'))
    model.add(Dense(n_classes, activation='softmax', name='Output'))

    return model
