'''
Import meta data into dataframe
'''

from pathlib import Path
import scipy.io as sio
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Read output file annotation
# Change the datapath here
def ImportMetaIntoDF(datapath, category, traintestratio):

    metaPath = Path.joinpath(datapath, category, r'cars_meta.mat')
    matContents = sio.loadmat(metaPath)
    metaLabels = np.ravel((matContents['class_names']).tolist())

    # Train images meta file read into a dataframe
    trainingDataMetaPath = Path.joinpath(datapath, category, r'cars_train_annos.mat')
    trainingDataMetaContents = sio.loadmat(trainingDataMetaPath)
    trainingData_df = pd.DataFrame();
    trainingDataFields = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
    for field in trainingDataFields:
        trainingData_df[field] = np.ravel((trainingDataMetaContents['annotations'][field]).tolist())
    trainingData_df['class']=trainingData_df['class'].astype(str)
    y = trainingData_df['class']

    # User sklearn train test split because it ensures class balanced
    # for e.g., we have 196 image classes, sklearn split will ensure
    # the train set has all 196 image classes input, whereas keras only
    # selects the last ratio of samples we specify
    train_df, test_df, y_train, y_test = train_test_split(trainingData_df, y, test_size=traintestratio)

    return train_df, test_df, metaLabels



