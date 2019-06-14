'''
This file will take dataframe, some parameters
and build generators out of them
'''
from fastai.vision import *

# data flow from dataframe
def data_split(dataframe, directory, imagesize, batchsize, testratio):

    # preprocessing function from fastai
    preprocess = get_transforms()
    
    src = (ImageList.from_df(dataframe, directory, cols='fname')).split_by_rand_pct(valid_pct=testratio, seed=42).label_from_df(cols='class')

    data = (src.transform(preprocess, size=imagesize)
            .databunch()
            .normalize(imagenet_stats))

    data.batch_size = batchsize

    return data
