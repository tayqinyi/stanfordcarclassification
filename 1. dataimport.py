'''
Import meta data, train and test data
'''

from pathlib import Path
import scipy.io as sio
import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications import densenet
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback

# Datapath, change it to the local data path
# Note, I extracted cars_test, cars_train and
# devkit folders into a folder 'data' same as the scripts
dataPath = Path(r'data')
TRAIN_FOLDER = Path(r'cars_train')
TEST_FOLDER = Path(r'cars_test')
MODEL = Path(r'models')
DEVKIT = Path(r'devkit')
BATCH_SIZE = 128
IMAGE_SIZE = 224
TRAIN_TEST_RATIO = 0.3
epochs = 10

# Read output file annotation
# Change the datapath here
metaPath = Path.joinpath(dataPath, DEVKIT, r'cars_meta.mat')
matContents = sio.loadmat(metaPath)
metaLabels = np.ravel((matContents['class_names']).tolist())

# Train images meta file read into a dataframe
trainMetaPath = Path.joinpath(dataPath, r'devkit\cars_train_annos.mat')
trainMetaContents = sio.loadmat(trainMetaPath)
train_df = pd.DataFrame();
trainFields = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
for field in trainFields:
    train_df[field] = np.ravel((trainMetaContents['annotations'][field]).tolist())
train_df['class']=train_df['class'].astype(str)

# data flow from dataframe
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   #shear_range=0.2,
                                   zoom_range=0.2,
                                   #fill_mode = 'constant',
                                   #cval = 1,
                                   rotation_range = 5,
                                   #width_shift_range=0.2,
                                   #height_shift_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=TRAIN_TEST_RATIO)

test_datagen = ImageDataGenerator(rescale=1/255,)

train_generator = train_datagen.flow_from_dataframe( dataframe=train_df,
                                                     directory=Path.joinpath(dataPath, TRAIN_FOLDER),
                                                     x_col='fname',
                                                     y_col='class',
                                                     target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                     class_mode='categorical',
                                                     batch_size=BATCH_SIZE,
                                                     subset='training')

validation_generator = train_datagen.flow_from_dataframe( dataframe=train_df,
                                                          directory=Path.joinpath(dataPath, TRAIN_FOLDER),
                                                          x_col='fname',
                                                          y_col='class',
                                                          target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                          class_mode='categorical',
                                                          batch_size=BATCH_SIZE,
                                                          subset='validation')


def build_model():
    base_model = densenet.DenseNet121(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                      include_top=False,
                                      pooling='avg')
    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = Dense(1000, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = Dense(500, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    predictions = Dense(len(metaLabels), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

model = build_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse'])
early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
callbacks_list = [early_stop, reduce_lr]

model_history = model.fit_generator(train_generator,
                                    steps_per_epoch=len(train_generator.filenames) // BATCH_SIZE,
                                    epochs=epochs,
                                    validation_data=validation_generator,
                                    validation_steps= len(train_generator.filenames) // BATCH_SIZE,
                                    callbacks=callbacks_list)