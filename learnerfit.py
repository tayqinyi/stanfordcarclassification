'''
fit a model based on the param
'''
import os
import time

def learn_fit(learn, modelpath, epochs, lr):
    learn.fit_one_cycle(epochs, lr)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    learn.save(file=os.path.join(modelpath, timestamp))