import pandas as pd
import os
import numpy as np


class DataLoader:
    """
    Class for loading and retrieving raw data from the CSVs into various formats.
    """

    def __init__(self):
        pass

    def load_all_three_dfs(self, max_size):

        train_df = self.load_data_df("random_split/train").sample(
            max_size, random_state=1
        )
        test_df = self.load_data_df("random_split/test").sample(
            max_size, random_state=1
        )
        val_df = self.load_data_df("random_split/dev").sample(max_size, random_state=1)

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
