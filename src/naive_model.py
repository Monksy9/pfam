import numpy as np
import pandas as pd


class NaiveModel:
    """
    Guesses top classes at random
    """

    def __init__(self, classes):
        self.classes = classes

    def predict(self, length):

        random_index = np.random.randint(low=0, high=len(self.classes), size=length)
        return pd.Series(self.classes[index] for index in random_index)
