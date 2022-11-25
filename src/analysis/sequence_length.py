import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", lambda x: "%.3f" % x)


class SequenceLengthExplorer:
    """Explorer object for reading in dataframes and creating plots relevant to sequence length."""

    def __init__(self, set_of_dfs: set, fold_names: str, top_n_families: int):
        self.list_of_dfs = list(map(self._create_sequence_length_column, set_of_dfs))
        self.fold_names = fold_names
        self.train_df = self.list_of_dfs[0]
        self.val_df = self.list_of_dfs[1]
        self.test_df = self.list_of_dfs[2]
        self.top_n_families = top_n_families
        self.top_n_families_list = self.retrieve_top_n_families(top_n_families)

    def plot_major_families_by_length(self):
        top_n_df = self.train_df[
            self.train_df["family_id"].isin(self.top_n_families_list)
        ]
        fig, ax = plt.subplots(1, 1, figsize=(30, 10))
        sns.histplot(
            data=top_n_df, x="sequence_length", hue="family_id", ax=ax, multiple="stack"
        )

    def plot_major_families_above_n_threshold(self, length_cutoff: int):
        long_sequences = self.train_df[self.train_df["sequence_length"] > length_cutoff]
        top_n_long_families_list = list(
            long_sequences["family_id"].value_counts().head(self.top_n_families).index
        )
        top_n_df = long_sequences[
            long_sequences["family_id"].isin(top_n_long_families_list)
        ]
        fig, ax = plt.subplots(1, 1, figsize=(30, 10))
        sns.histplot(
            data=top_n_df, x="sequence_length", hue="family_id", ax=ax, multiple="stack"
        )

    def print_sequence_length(self):
        self._print_length_percentiles()
        self._plot_sequence_length_by_fold()

    def retrieve_top_n_families(self, top_n_families):
        return list(
            self.train_df["family_id"].value_counts().head(top_n_families).index
        )

    def _create_sequence_length_column(self, df: pd.DataFrame):
        df["sequence_length"] = df["sequence"].apply(lambda x: len(x))
        return df

    def _print_length_percentiles(self):
        print(
            self.train_df["sequence_length"].describe(
                percentiles=[0.05, 0.1, 0.9, 0.95]
            )
        )

    def _plot_sequence_length_by_fold(self):
        fig, ax = plt.subplots(1, 3, figsize=(30, 10))

        for fold in range(3):
            self._plot_sequence_length_df(
                self.list_of_dfs[fold], self.fold_names[fold], ax[fold]
            )
            self._plot_reference_line(ax[fold])

    def _plot_sequence_length_df(self, df: pd.DataFrame, fold_name: str, ax: plt.Axes):
        sns.histplot(df["sequence_length"], ax=ax, kde=True, color="b").set_title(
            "Sequence length " + str(fold_name)
        )

    def _plot_reference_line(self, ax: plt.Axes, seqeunce_length_cutoff: int = 512):
        ax.axvline(x=seqeunce_length_cutoff, color="r")
