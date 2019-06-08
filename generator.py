'''
This file will take dataframe, some parameters
and build generators out of them
'''

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.densenet import preprocess_input

# data flow from dataframe
def BuildGeneratorFromDF(dataframe, directory, splitratio, imagesize, batchsize):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       preprocessing_function=preprocess_input,
                                       validation_split=splitratio)

    test_datagen = ImageDataGenerator(rescale=1/255,)

    train_generator = train_datagen.flow_from_dataframe( dataframe=dataframe,
                                                         directory=directory,
                                                         x_col='fname',
                                                         y_col='class',
                                                         target_size=(imagesize, imagesize),
                                                         class_mode='categorical',
                                                         batch_size=batchsize,
                                                         subset='training')

    validation_generator = train_datagen.flow_from_dataframe( dataframe=dataframe,
                                                              directory=directory,
                                                              x_col='fname',
                                                              y_col='class',
                                                              target_size=(imagesize, imagesize),
                                                              class_mode='categorical',
                                                              batch_size=batchsize,
                                                              subset='validation')

    return train_generator, validation_generator
