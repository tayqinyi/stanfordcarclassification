'''
This file will take dataframe, some parameters
and build generators out of them
'''

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# data flow from dataframe
def BuildGeneratorFromDF(dataframe, directory, splitratio, imagesize, batchsize):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       #shear_range=0.2,
                                       zoom_range=0.2,
                                       #fill_mode = 'constant',
                                       #cval = 1,
                                       rotation_range = 5,
                                       #width_shift_range=0.2,
                                       #height_shift_range=0.2,
                                       horizontal_flip=True,
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
