'''
This file will take dataframe, some parameters
and build generators out of them
'''
from fastai.vision import *

# data flow from dataframe
def data_split(train_df, test_df, train_dir, test_dir, imagesize, batchsize, testratio):

    # preprocessing function from fastai
    preprocess = get_transforms()

    test = ImageList.from_df(test_df, test_dir, cols='fname')
    train = (ImageList.from_df(train_df, train_dir, cols='fname')).split_by_rand_pct(valid_pct=testratio, seed=42).label_from_df(cols='class').add_test(test)

    data = (train.transform(preprocess, size=imagesize)
            .databunch()
            .normalize(imagenet_stats))

    data.batch_size = batchsize

    return data
