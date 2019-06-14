'''
fit a model based on the param
'''
from fastai.utils.mem import *

def learn_fit(learn, epochs):

    lr = 1e-3
    learn.fit_one_cycle(epochs, max_lr = lr)