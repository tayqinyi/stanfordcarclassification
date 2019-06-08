'''
Import meta data into dataframe
'''

from pathlib import Path
import scipy.io as sio
import pandas as pd
import numpy as np

TRAIN_TEST_RATIO = 0.3
# Read output file annotation
# Change the datapath here
def ImportMetaIntoDF(datapath, category):

    metaPath = Path.joinpath(datapath, category, r'cars_meta.mat')
    matContents = sio.loadmat(metaPath)
    metaLabels = np.ravel((matContents['class_names']).tolist())

    # Train images meta file read into a dataframe
    trainMetaPath = Path.joinpath(datapath, category, r'cars_train_annos.mat')
    trainMetaContents = sio.loadmat(trainMetaPath)
    train_df = pd.DataFrame();
    trainFields = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
    for field in trainFields:
        train_df[field] = np.ravel((trainMetaContents['annotations'][field]).tolist())
    train_df['class']=train_df['class'].astype(str)

    return train_df, metaLabels



