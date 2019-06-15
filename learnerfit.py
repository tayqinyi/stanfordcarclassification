'''
fit a model based on the param
'''
import os
import time

def learn_fit(learn, epochs, stage, modelpath):
    lr = 1e-3
    learn.fit_one_cycle(epochs, max_lr = lr)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    learn.save(file=os.path.join(modelpath, timestamp))