'''
This file will take dataframe, some parameters
and build generators out of them
'''

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.densenet import preprocess_input

# data flow from dataframe
def BuildGeneratorFromDF(dataframe, directory, imagesize, batchsize):
    datagen = ImageDataGenerator(rescale=1. / 255,
                                 preprocessing_function=preprocess_input)

    generator = datagen.flow_from_dataframe( dataframe=dataframe,
                                             directory=directory,
                                             x_col='fname',
                                             y_col='class',
                                             target_size=(imagesize, imagesize),
                                             class_mode='categorical',
                                             batch_size=batchsize)

    return generator
