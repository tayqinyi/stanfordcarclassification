'''
Import meta data, train and test data
'''

from pathlib import Path
import scipy.io as sio
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

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
train_df['fname'] = [str(Path.joinpath(dataPath, TRAIN_FOLDER, fname)) for fname in train_df['fname']]

# Train test split
train_features, test_features, train_labels, test_labels = train_test_split(train_df[['fname', 'bbox_x1', 'bbox_y1', 'width', 'height']],
                                                                            train_df['class'],  # labels
                                                                            train_size=0.7,
                                                                            random_state=0)     # randomize the data

def import_image(filename, label, box):
    box = tf.cast(box, tf.int32);
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.per_image_standardization(image)
    image = tf.image.crop_to_bounding_box(image, box[1], box[0],
                                          box[3], box[2]) # Crop according to the box coordinates in mat file
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    return image, label

train_data = tf.data.Dataset.from_tensor_slices((tf.constant(train_features['fname']),
                                                 tf.constant(train_labels),
                                                 tf.constant(train_features[['bbox_x1', 'bbox_y1', 'width', 'height']].values)))\
    .map(import_image).shuffle(buffer_size=1000).batch(BATCH_SIZE)

test_data = tf.data.Dataset.from_tensor_slices((tf.constant(test_features['fname']),
                                                tf.constant(test_labels),
                                                tf.constant(test_features[['bbox_x1', 'bbox_y1', 'width', 'height']].values)))\
    .map(import_image).shuffle(buffer_size=1000).batch(BATCH_SIZE)

'''
Models
'''
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
prediction_layer = tf.keras.layers.Dense(1, activation='softmax')
model = tf.keras.Sequential([
    base_model,
    maxpool_layer,
    prediction_layer
])

learning_rate = 0.0001

model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
             loss='categorical_crossentropy',
             metrics=['accuracy'])

'''
Train
'''
num_epochs = 2
steps_per_epoch = round(len(train_labels))//BATCH_SIZE
val_steps = 20

history = model.fit(train_data.repeat(),
                    epochs=num_epochs,
                    steps_per_epoch = steps_per_epoch,
                    validation_data=test_data.repeat(),
                    validation_steps=val_steps)