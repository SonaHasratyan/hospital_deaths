import numpy as np
import pandas as pd

"""
    This file can have 2 arguments.
    1.  --data_path: absolute path to a train or test dataset.
    2.  --inference: whether to run model in a training mode or in testing mode. 
        a.  training mode (by default - If the argument is not given)
        b.  testing mode (pass manually as an argument) -  When running in testing mode, outputs are saved in 
            predictions.json file, which has 2 keys:
            1)  predict_probas: predicted probabilities for each test datapoint
            2)  threshold: the best threshold recommended to take
                (all probabilities higher than threshold would be converted to 1, otherwise to 0).
"""


class Pipeline:
    """
       Has a run method.
    """

    def __init__(self):
        pass

    def run(self, X: np.ndarray, test: bool):
        pass
