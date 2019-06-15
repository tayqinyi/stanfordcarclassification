'''
Import meta data into dataframe
'''
import os
import scipy.io as sio
import pandas as pd
import numpy as np
from PIL import Image

# Read output file annotation
# Change the datapath here
def ImportMetaIntoDF(devkit, train_path, test_path, cropped_train_path, cropped_test_path, crop):

    # Train images meta file read into a dataframe
    train_data_df, test_data_df = pd.DataFrame(), pd.DataFrame();
    train_annos_path = os.path.join(devkit, r'cars_train_annos.mat')
    test_annos_path = os.path.join(devkit, r'cars_test_annos.mat')

    train_meta_contents = sio.loadmat(train_annos_path)
    test_meta_contents = sio.loadmat(test_annos_path)
    test_data_fields = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'fname']
    train_data_fields = test_data_fields.copy().append('class')

    for field in test_data_fields:
        train_data_df[field] = np.ravel((train_meta_contents['annotations'][field]).tolist())
        test_data_df[field] = np.ravel((test_meta_contents['annotations'][field]).tolist())
    train_data_df['class'] = np.ravel((train_meta_contents['annotations']['class']).tolist())
    train_data_df['class']=train_data_df['class'].astype(str)

    # Crop images according to the annos
    if(crop):
        if(not os.path.exists(cropped_train_path)):
            os.makedirs(cropped_train_path, exist_ok=True)
        if(not os.path.exists(cropped_test_path)):
            os.makedirs(cropped_test_path, exist_ok=True)

        # Crop train images
        for index, row in train_data_df.iterrows():
            source_path = os.path.join(train_path, row['fname'])
            image = Image.open(source_path)
            image = image.crop((row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']))
            target_path = os.path.join(cropped_train_path, row['fname'])
            image.save(target_path)

        # Crop test images
        for index, row in test_data_df.iterrows():
            source_path = os.path.join(test_path, row['fname'])
            image = Image.open(source_path)
            image = image.crop((row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']))
            target_path = os.path.join(cropped_test_path, row['fname'])
            image.save(target_path)

    return train_data_df, test_data_df

