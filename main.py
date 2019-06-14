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
  - learnerbuild.py
  - main.py

You can then run this script, this script will
1. Import previously download data as data frame
2. Prepare an Image generator to flow into model
3. Build a Densenet model
4. Fit the model
5. Predict the images in the cars_test folder
'''

import dataimport, datasplit, learnerbuild, learnerfit
from pathlib import Path
import argparse, sys

# Datapath, change it to the local data path
# Note, I extracted cars_test, cars_train and
# devkit folders into a folder 'data' same as the scripts
DATA_PATH = Path(r'data')
TRAIN_FOLDER = Path.joinpath(DATA_PATH, r'cars_train')
TEST_FOLDER = Path.joinpath(DATA_PATH, r'cars_test')
CROPPED_TRAIN_FOLDER = Path.joinpath(DATA_PATH, r'cropped_cars_train')
CROPPED_TEST_FOLDER = Path.joinpath(DATA_PATH, r'cropped_cars_test')
MODEL = Path(r'models')
DEVKIT = Path.joinpath(DATA_PATH, r'devkit')
BATCH_SIZE = 8
IMAGE_SIZE = 224
TEST_RATIO = 0.2
EPOCHS = 30


def main(args):

    # 1. Import data into dataframe (filename, class) and also the meta labels
    train_df = dataimport.ImportMetaIntoDF(DEVKIT, TRAIN_FOLDER, CROPPED_TRAIN_FOLDER, args.crop)
    # number of classes from the metalabels list taken from cars_train_annos
    n_class = len(train_df['class'])

    # 2. Build data split for generator to flow and train test split
    data = datasplit.data_split(train_df,
                                CROPPED_TRAIN_FOLDER,
                                imagesize=224,
                                batchsize=BATCH_SIZE,
                                testratio=TEST_RATIO)

    # 3. Build a model based on specifications
    learner = learnerbuild.build_learn(data)

    # 4. Fit the model
    learnerfit.learn_fit(learner, EPOCHS)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--crop',
                        default=False)

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))