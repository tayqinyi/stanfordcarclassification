'''
Build a model
'''
from fastai.vision import *
from fastai.callbacks import EarlyStoppingCallback

def build_learn(data):
    # Use pretrained resnet152
    arch = models.resnet152
    # Prevent overfitting callback
    callbacks = [partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.01, patience=3)]
    learn = cnn_learner(data, arch, metrics=[error_rate], callback_fns=callbacks)

    return learn
