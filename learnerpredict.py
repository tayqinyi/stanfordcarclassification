# This file will run prediction on the test dataset
# in the learner object
import numpy as np
from fastai.basic_data import DatasetType

def predict(learner):

    # Probabilities will be in the format of probability for each label
    # tensor([[3.8513e-06, 1.2691e-07, 2.5590e-05, ..., 1.7225e-05, 7.4509e-06,
    #          1.1635e-05],
    #         [6.5386e-07, 4.5803e-07, 1.7359e-07, ..., 1.4434e-08, 1.0845e-08,
    #          6.4887e-08],
    #         ...,
    #         [7.2411e-08, 3.1668e-06, 3.6676e-08, ..., 7.1346e-07, 3.6845e-07,
    #          2.1331e-08],
    #         [9.0684e-08, 7.6374e-08, 3.8296e-08, ..., 9.3261e-08, 7.8992e-06,
    #          1.0119e-06]])
    probabilities, *_ = learner.get_preds(DatasetType.Test)
    # We will use the following code to extract the position of the highest probability
    labels_locs = np.argmax(probabilities, 1)
    # However, note that we have only extracted the positions, the exact classes
    # have to be retrieved from the classes list
    predictions = [learner.data.classes[int(x)] for x in labels_locs]

    return predictions