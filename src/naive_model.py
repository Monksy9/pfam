import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


class NaiveModel:
    """
    Class for reading in a train dataframe to select classes from randomly, and broadcasts to size of test set. Random Guesser.
    """

    def __init__(self):
        pass

    def predict(self, length):

        random_index = np.random.randint(low=0, high=len(self.classes), size=length)
        return pd.Series(self.classes[index] for index in random_index)

    def naive_model_accuracy(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Takes in a train dataframe to select classes from randomly, and broadcasts to size of test set."""
        self.classes = list(train_df["family_accession"].value_counts().head(10).index)
        naive_preds = self.predict(len(test_df))
        print(
            "Accuracy of Naive method "
            + str(round(accuracy_score(test_df["family_accession"], naive_preds), 3))
        )
