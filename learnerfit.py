'''
fit a model based on the param
'''
from pathlib import Path

def learn_fit(learn, epochs, stage, modelpath):
    lr = 1e-3
    learn.fit_one_cycle(epochs, max_lr = lr)
    learn.save(file=Path.joinpath(modelpath, 'stage - ' + str(stage)))