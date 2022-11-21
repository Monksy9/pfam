import pandas as pd


def SimpleEmbedder():
    """
    Takes a list of embeddings and returns the percentage of each amino acid
    """

    def __init__(self):
        self.list_of_acids = self.read_acids()

    def read_acids(self):
        with open("acids.txt") as f:
            acids = list(f.read())
        return acids

    def obtain_pct_features(self, input_df):
        dict_of_percentages = {}
        for acid in self.list_of_acids:
            dict_of_percentages[acid] = input_df["sequence"].apply(
                lambda x: x.count(acid) / len(x)
            )
        df = pd.DataFrame(dict_of_percentages)
        return df
