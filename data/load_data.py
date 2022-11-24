import pandas as pd
import os
import numpy as np


class DataLoader:
    """
    Class for loading and retrieving raw data from the CSVs into various formats.
    """

    def __init__(self):
        pass

    def load_all_three_dfs(self, train_size=None, val_size=None, test_size=None):
        
        train_df = self.load_data_df("random_split/train")
        test_df = self.load_data_df("random_split/test")
        val_df = self.load_data_df("random_split/dev")

        if train_size:
            train_df = train_df.sample(train_size, random_state=1)
        if test_size:
            test_df = test_df.sample(test_size, random_state=1)
        if val_size:
            val_df = val_df.sample(val_size, random_state=1)

        return train_df, val_df, test_df

    def load_data_df(self, path):

        full_path = [path + "/" + file for file in os.listdir(path=path)]

        list_of_dfs = []
        for file in full_path:
            list_of_dfs.append(self.read_file_df(file))

        return pd.concat(list_of_dfs)

    def read_file_df(self, file):

        return pd.read_csv(file, index_col=None)

    def numpy_input_sequence(self, df):
        return np.array(df["sequence"].values)


if __name__ == "__main__":
    Loader = DataLoader()

    df = Loader.load_data_df("random_split/train")

    print(df.shape)
