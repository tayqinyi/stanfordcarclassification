'''
Import meta data, train and test data
'''

from pathlib import Path
import scipy.io as sio
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from matplotlib import pyplot
# Datapath, change it to the local data path
# Note, I extracted cars_test, cars_train and
# devkit folders into a folder 'data' same as the scripts
dataPath = Path(r'data')
TRAIN_FOLDER = Path(r'cars_train')
TEST_FOLDER = Path(r'cars_test')
DEVKIT = Path(r'devkit')
BATCH_SIZE = 128
IMAGE_SIZE = 100

# Read output file annotation
# Change the datapath here
metaPath = Path.joinpath(dataPath, DEVKIT, r'cars_meta.mat')
matContents = sio.loadmat(metaPath)
metaLabels = np.ravel((matContents['class_names']).tolist())

# Train images meta file read into a dataframe
trainMetaPath = Path.joinpath(dataPath, r'devkit\cars_train_annos.mat')
trainMetaContents = sio.loadmat(trainMetaPath)
train_df = pd.DataFrame();
trainFields = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
for field in trainFields:
    train_df[field] = np.ravel((trainMetaContents['annotations'][field]).tolist())

# Image width, height
train_df['width'], train_df['height'] = train_df['bbox_x2'] - train_df['bbox_x1'], train_df['bbox_y2'] - train_df['bbox_y1']
train_df = train_df.applymap(str)
train_df['fname'] = [str(Path.joinpath(dataPath, TRAIN_FOLDER, fname)) for fname in train_df['fname']]

# Train test split
train_features, test_features, train_labels, test_labels = train_test_split(train_df[['fname', 'bbox_x1', 'bbox_y1', 'width', 'height']],
                                                                            train_df['class'],  # labels
                                                                            train_size=0.7,
                                                                            random_state=0)     # randomize the data

training_df: pd.DataFrame = pd.DataFrame(
    data={
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10),
        'feature3': np.random.rand(10),
        'target': np.random.randint(0, 3, 10)
    }
)
features = ['feature1', 'feature2', 'feature3']
training_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(training_df[features].values, tf.string),
            tf.cast(training_df['target'].values, tf.int32)
        )
    )
)

def import_image(feature, label):
    image = tf.io.read_file(feature[0])
    image = tf.image.decode_jpeg(image)
    image = tf.image.per_image_standardization(image)
    image = tf.image.crop_to_bounding_box(image, tf.cast(feature[2], tf.uint32), tf.cast(feature[1], tf.uint32),
                                          tf.cast(feature[4], tf.uint32), tf.cast(feature[3], tf.uint32)) # Crop according to the box coordinates in mat file
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    return image, label

train_data = tf.data.Dataset.from_tensor_slices((tf.constant(train_features.values), tf.constant(train_labels)))\
    .map(import_image).shuffle(buffer_size=10000).batch(BATCH_SIZE)
train_data = tf.data.Dataset.from_tensor_slices((tf.constant(test_features.values), tf.constant(test_labels)))\
    .map(import_image).shuffle(buffer_size=10000).batch(BATCH_SIZE)



