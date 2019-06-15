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
3. Get a pretrained resnet152 model, attach layers in the back
4. Fit the model
5. Predict the images in the cars_test folder
'''

import dataimport, datasplit, learnerbuild, learnerfit, learnerpredict
import os
import argparse, sys
import time

# Datapath, change it to the local data path
# Note, I extracted cars_test, cars_train and
# devkit folders into a folder 'data' same as the scripts
DATA_PATH = r'data'
TRAIN_FOLDER = os.path.join(DATA_PATH, r'cars_train')
TEST_FOLDER = os.path.join(DATA_PATH, r'cars_test')
CROPPED_TRAIN_FOLDER = os.path.join(DATA_PATH, r'cropped_cars_train')
CROPPED_TEST_FOLDER = os.path.join(DATA_PATH, r'cropped_cars_test')
MODEL_PATH = r'..\..\..\models'
MODELS_PATH = r'models'
DEVKIT = os.path.join(DATA_PATH, r'devkit')
PREDICTION_PATH = r'predictions'
BATCH_SIZE = 8
IMAGE_SIZE = 224
TEST_RATIO = 0.2
EPOCHS = 30


def main(args):

    args = parse_arguments(sys.argv[1:])

    '''
    Part 1. Data import, processing
    '''
    # 1. Import data into dataframe (filename, class) and also the meta labels
    train_df, test_df = dataimport.ImportMetaIntoDF(DEVKIT, TRAIN_FOLDER, TEST_FOLDER, CROPPED_TRAIN_FOLDER, CROPPED_TEST_FOLDER, args.crop)

    # 2. Build data with datasplit inside
    # Data has 3 components:
    # train data: 80% of the data in the train folder
    # validation data: 20% of the data in the train folder
    # test data: all of the data in the test folder
    data = datasplit.data_split(train_df,
                                test_df,
                                CROPPED_TRAIN_FOLDER,
                                CROPPED_TEST_FOLDER,
                                imagesize=224,
                                batchsize=BATCH_SIZE,
                                testratio=TEST_RATIO
                                )

    '''
    Part 2. Construct a model
    '''
    # 3. Build a model
    learner = learnerbuild.build_learn(data)


    '''
    Part 3. Fit or load a model we have trained, fit will also execute finetune
    '''
    # 4. Fit if specified
    if (args.train):
        # fit the model for the first time
        learnerfit.learn_fit(learner, EPOCHS, 1, MODEL_PATH)
    else:
        # load the latest model
        files = os.listdir(MODEL_PATH)
        paths = [os.path.join(MODEL_PATH, basename) for basename in files]
        path = max(paths, key=os.path.getctime)
        learner.load(file=path)

    # 5. Fine tune by adjusting learning rate
    #learner.unfreeze()
    #learner.fit_one_cycle(2, max_lr=slice(1e-5, 1e-4))

    '''
    Part 4. Do prediction on the test data and output into a text file
    '''
    # 6. Predict
    predictions = learnerpredict.predict(learner)
    # write result to the predictions folder
    timestamp = time.strftime("%Y%m%d-%H%M%S") + '.txt'
    path = os.path.join(PREDICTION_PATH, timestamp)
    with open(path, "w") as output:
        for i in predictions:
            output.write(i + '\n')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--train',
                        default=False,
                        help='Specify whether we want to train the model with train data, if False, we will try to load a pretrained model in the models folder')

    parser.add_argument('--crop',
                        default=False,
                        help='Specify whether we want to crop the images during data import, should only specify this as true first time running this script')

    return parser.parse_args(argv)


if __name__ == "__main__":
    #main(parse_arguments(sys.argv[1:])
    args = parse_arguments(sys.argv[1:])
    print(args)
    '''
    Part 1. Data import, processing
    '''
    print('============== Part 1 Data import, processing ==============', args)
    # Import data into dataframe (filename, class) and also the meta labels
    train_df, test_df = dataimport.ImportMetaIntoDF(DEVKIT, TRAIN_FOLDER, TEST_FOLDER, CROPPED_TRAIN_FOLDER, CROPPED_TEST_FOLDER, args.crop)

    # Build data with datasplit inside
    # Data has 3 components:
    # train data: 80% of the data in the train folder
    # validation data: 20% of the data in the train folder
    # test data: all of the data in the test folder
    data = datasplit.data_split(train_df,
                                test_df,
                                CROPPED_TRAIN_FOLDER,
                                CROPPED_TEST_FOLDER,
                                imagesize=224,
                                batchsize=BATCH_SIZE,
                                testratio=TEST_RATIO
                                )
    print('============== Part 1 Data import, processing done! ==============', args)

    print('============== Part 2 Model building start ==============', args)
    '''
    Part 2. Construct a model
    '''
    # Build a model
    learner = learnerbuild.build_learn(data)
    print('============== Part 2 Model building done! ==============', args)


    '''
    Part 3. Fit or load a model we have trained, fit will also execute finetune
    '''
    if(args.train):
        print('============== Part 3 Fit model start ==============', args)
    else:
        print('============== Part 3 Train = False ==> Load latest model start ==============', args)
    # Fit if specified
    if (args.train):
        # fit the model for the first time
        learnerfit.learn_fit(learner, EPOCHS, 1, MODEL_PATH)
    else:
        # load the latest model
        files = os.listdir(MODELS_PATH)
        paths = [os.path.join(MODELS_PATH, basename) for basename in files]
        path = os.path.splitext(max(paths, key=os.path.getctime))[0]
        learner.load(file=os.path.join(r'..\..\..', path))
    if(args.train):
        print('============== Part 3 Fit model done! ==============', args)
    else:
        print('============== Part 3 Train = False ==> Load latest model done! ==============', args)

    # Fine tune by adjusting learning rate
    #learner.unfreeze()
    #learner.fit_one_cycle(2, max_lr=slice(1e-5, 1e-4))

    '''
    Part 4. Do prediction on the test data and output into a text file
    '''
    print('============== Part 4 Prediction start ==============', args)
    # Predict
    predictions = learnerpredict.predict(learner)
    # write result to the predictions folder
    timestamp = time.strftime("%Y%m%d-%H%M%S") + '.txt'
    path = os.path.join(PREDICTION_PATH, timestamp)
    with open(path, "w") as output:
        for i in predictions:
            output.write(i + '\n')
    print('============== Part 4 Predicting done! ==============', args)
    print('============== All done! Please view the latest prediction output in the predictions folder ==============', args)