'''
This script uses transfer learning on a pretrained CNN (Densenet)
for standford cars image classification.
The data can be downloaded from:
https://ai.stanford.edu/~jkrause/cars/car_dataset.html

To get this script to work
- Download three files: Training images, Testing images and DevKit
- Extract them into a data folder in the same root as the script, like this:
  - data
    - cars_test
    - cars_train
    -devkit
  - dataimport.py
  - model.py
  - main.py

You can then run this script, this script will
1. Import previously download data as data frame
2. Prepare an Image generator to flow into model
3. Build a Densenet model
4. Fit the model
5. Predict the images in the cars_test folder
'''

import dataimport, generator, model, modelfit
from pathlib import Path

# Datapath, change it to the local data path
# Note, I extracted cars_test, cars_train and
# devkit folders into a folder 'data' same as the scripts
DATA_PATH = Path(r'data')
TRAIN_FOLDER = Path(r'cars_train')
TEST_FOLDER = Path(r'cars_test')
MODEL = Path(r'models')
DEVKIT = Path(r'devkit')
BATCH_SIZE = 32
IMAGE_SIZE = 224
TRAIN_TEST_RATIO = 0.25
EPOCHS = 50

'''
1. Import data into dataframe (filename, class) and also the meta labels
'''
train_df, test_df, meta_labels = dataimport.ImportMetaIntoDF(DATA_PATH, DEVKIT, TRAIN_TEST_RATIO)
# number of classes from the metalabels list taken from cars_train_annos
n_class=len(meta_labels)

'''
2. Build generator for keras to flow data based on the filename in the dataframe
'''
train_generator = generator.BuildGeneratorFromDF(train_df,
                                                 Path.joinpath(DATA_PATH, TRAIN_FOLDER),
                                                 imagesize=224,
                                                 batchsize=BATCH_SIZE)

validation_generator = generator.BuildGeneratorFromDF(test_df,
                                                      Path.joinpath(DATA_PATH, TRAIN_FOLDER),
                                                      imagesize=224,
                                                      batchsize=BATCH_SIZE)


'''
3. Build a model based on specifications
'''
model = model.build_model(IMAGE_SIZE, n_class)

'''
4. Fit the model
'''
model, model_history = modelfit.fit_model(model, train_generator, validation_generator, EPOCHS, BATCH_SIZE)