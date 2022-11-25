import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import sequence_length as sl

style.use("seaborn-poster")
style.use("ggplot")
pd.option_context(
    "display.max_rows",
    None,
    "display.max_columns",
    None,
    "display.precision",
    3,
)


class PfamExplorer:
    """
    Top level class responsible for taking in three folds and producing logging/charts on key information.
    Presents only top n families in certain charts for convenience.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        top_n_families=10,
    ):
        self.set_of_dfs = (train_df, val_df, test_df)
        self.fold_names = ["train", "val", "test"]
        self.top_n_families = top_n_families
        self.sequence_length_explorer = sl.SequenceLengthExplorer(
            self.set_of_dfs, self.fold_names, top_n_families
        )

    def print_family_imbalance_details(self):
        train_top_n_families_pct = (
            self._retrieve_top_n_df(self.set_of_dfs[0]).shape[0]
            / self.set_of_dfs[0].shape[0]
        ) * 100
        test_top_n_families_pct = (
            self._retrieve_top_n_df(self.set_of_dfs[2]).shape[0]
            / self.set_of_dfs[2].shape[0]
        ) * 100

        print(
            self.set_of_dfs[0]["family_id"].value_counts().head(5)
            / len(self.set_of_dfs[0])
        )

        print(
            "The top "
            + str(self.top_n_families)
            + " family IDs represent %.2f" % train_top_n_families_pct
            + "pct of train"
        )

        print(
            "The top "
            + str(self.top_n_families)
            + " family IDs represent %.2f" % test_top_n_families_pct
            + "pct of test"
        )

    def family_summary_plots(self):
        self._plot_countplots()

    def print_n_rows_all_folds(self):
        for i in range(3):
            self._print_n_rows_df(self.set_of_dfs[i], self.fold_names[i])

    def print_sequence_length_info(self):
        print("Sequence length percentiles: ")
        self.sequence_length_explorer.print_sequence_length()
        print("\n\n")

    def plot_major_families(self):
        self.sequence_length_explorer.plot_major_families_by_length()

    def plot_major_families_above_threshold(self, length_cutoff):
        self.sequence_length_explorer.plot_major_families_above_n_threshold(
            length_cutoff=length_cutoff
        )

    def family_summary_text(self):
        for i in range(3):
            self._unique_families_per_df(self.set_of_dfs[i], self.fold_names[i])

        self.check_for_unseen_families()

        self._counts_per_family()

    def multiple_records_per_sequence(self):
        counts = self.set_of_dfs[0].groupby("sequence", as_index=False).count()

        more_than_one_sequence = counts[counts["family_accession"] > 1][
            ["sequence", "sequence_name"]
        ].sort_values(by="sequence_name", ascending=False)
        print(
            "There are "
            + str(more_than_one_sequence["sequence"].nunique())
            + " unique sequences, that have multiple family_ids "
        )
        print(
            "There are "
            + str(more_than_one_sequence["sequence_name"].sum())
            + " rows in total that have multiple family ids \n"
        )

        print("Here's an example:")

        most_repeated = more_than_one_sequence["sequence"].values[0]

        print(
            self.set_of_dfs[0][self.set_of_dfs[0]["sequence"] == most_repeated][
                ["sequence", "sequence_name"]
            ].head(5)
        )

        print("\n\n")

    def check_for_unseen_families(self):
        list_of_families = list(map(self._number_of_families, self.set_of_dfs))

        new_test_families = self._find_difference_of_families(
            list_of_families[2], list_of_families[0]
        )
        new_val_families = self._find_difference_of_families(
            list_of_families[1], list_of_families[0]
        )

        print(
            "There are "
            + str(len(new_test_families) + len(new_val_families))
            + " new families in test / val vs train"
        )

    def _number_of_families(self, df):
        return list(df["family_accession"].unique())

    def _print_n_rows_df(self, df, fold_name):
        print("Number of rows: " + str(df.shape[0]) + " for fold: " + str(fold_name))

    def _unique_families_per_df(self, df, fold_name):
        no_families = df["family_accession"].nunique()
        print("There are " + str(no_families) + " unique families in " + str(fold_name))

    def _find_difference_of_families(self, df_1, df_2):
        return [family for family in df_1 if family not in df_2]

    def _plot_countplots(self, top_n=10):

        fig, ax = plt.subplots(1, 1, figsize=(30, 15))
        list_of_top_n_folds = list(map(self._retrieve_top_n_df, self.set_of_dfs))
        fold_names = ["train", "test", "val"]
        all_folds_top_n = pd.concat(list_of_top_n_folds, ignore_index=True)
        all_folds_top_n["fold"] = np.repeat(
            fold_names, list(map(len, list_of_top_n_folds))
        )
        sns.countplot(
            x=all_folds_top_n["family_id"],
            order=all_folds_top_n["family_id"].value_counts().index,
            hue=all_folds_top_n["fold"],
        )
        title = "Top " + str(top_n) + " families across train, test and val"
        ax.set_title(title)

    def _retrieve_top_n_df(self, df):
        return df[
            df["family_id"].isin(
                self.sequence_length_explorer.retrieve_top_n_families(
                    self.top_n_families
                )
            )
        ]

    def _counts_per_family(self):

        print("Number of rows per family:")
        print(self.set_of_dfs[0]["family_accession"].value_counts().describe())
        print("\n\n")
