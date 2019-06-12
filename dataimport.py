'''
Import meta data into dataframe
'''
import os
from pathlib import Path
import scipy.io as sio
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

# Read output file annotation
# Change the datapath here
def ImportMetaIntoDF(devkit, img_path, cropped_img_path, traintestratio, crop):

    meta_path = Path.joinpath(devkit, r'cars_meta.mat')
    mat_contents = sio.loadmat(meta_path)
    meta_labels = np.ravel((mat_contents['class_names']).tolist())

    # Train images meta file read into a dataframe
    annos_path = Path.joinpath(devkit, r'cars_train_annos.mat')
    training_data_meta_contents = sio.loadmat(annos_path)
    training_data_df = pd.DataFrame();
    training_data_fields = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
    for field in training_data_fields:
        training_data_df[field] = np.ravel((training_data_meta_contents['annotations'][field]).tolist())
    training_data_df['class']=training_data_df['class'].astype(str)
    y = training_data_df['class']

    # Crop images according to the annos
    if(crop):
        if(not os.path.exists(cropped_img_path)):
            os.makedirs(cropped_img_path, exist_ok=True)

        for index, row in training_data_df.iterrows():
            source_path = Path.joinpath(img_path, row['fname'])
            image = Image.open(source_path)
            image = image.crop((row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']))
            target_path = Path.joinpath(cropped_img_path, row['fname'])
            image.save(target_path)

    # User sklearn train test split because it ensures class balanced
    # for e.g., we have 196 image classes, sklearn split will ensure
    # the train set has all 196 image classes input, whereas keras only
    # selects the last ratio of samples we specify
    train_df, test_df, y_train, y_test = train_test_split(training_data_df, y, test_size=traintestratio)


    return train_df, test_df, meta_labels

