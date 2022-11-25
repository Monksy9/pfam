import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor


class TrainModel:
    """Class responsible for loading data, training the end classifier"""

    def __init__(
        self,
        list_of_embeddings_to_use,
        num_gpus,
        train_time_limit_seconds,
        eval_metric,
        label_count_threshold,
        verbosity=2,
    ):

        self.list_of_embeddings_to_use = list_of_embeddings_to_use

        self.tabular_predictor = TabularPredictor(
            label="family_accession",
            problem_type="multiclass",
            eval_metric=eval_metric,
            learner_kwargs={"label_count_threshold": label_count_threshold},
            verbosity=verbosity,
        )
        self.num_gpus = num_gpus
        self.time_limit = train_time_limit_seconds

    def load_data(self):

        dict_of_features = {"train": [], "test": [], "val": []}
        dict_of_columns = {}
        for embedder in self.list_of_embeddings_to_use:
            path = "embeddings/" + str(embedder) + "/"
            for fold in ["train", "test", "val"]:
                df = self._load_df(path + fold + ".csv")
                dict_of_features[fold].append(df)
                dict_of_columns[embedder] = df.columns

        train_df, val_df, test_df = self._combine_all_embeddings_all_folds(
            dict_of_features
        )

        return dict_of_columns, train_df, val_df, test_df

    def train_model(self, set_of_dfs):

        list_of_tabular_datasets = self.convert_df_to_tabular(set_of_dfs=set_of_dfs)

        list_of_tabular_datasets_wo_sequence = [
            dataset.drop("sequence", axis=1) for dataset in list_of_tabular_datasets
        ]
        list_of_tabular_datasets_wo_sequence_and_family_accession = [
            dataset.drop(["sequence", "family_accession"], axis=1)
            for dataset in list_of_tabular_datasets
        ]

        self.tabular_predictor.fit(
            list_of_tabular_datasets_wo_sequence[0],
            num_gpus=self.num_gpus,
            tuning_data=list_of_tabular_datasets_wo_sequence[1],
            time_limit=self.time_limit,
        )

    def autogluon_predict(self, test_tabular: TabularDataset):

        assert "sequence" not in test_tabular.columns
        assert "family_accession" not in test_tabular.columns

        return self.tabular_predictor.predict(test_tabular)

    def return_results(self):
        return self.tabular_predictor.fit_summary(show_plot=True)

    def evaluate_test_set(self, test_df: pd.DataFrame):

        tabular_test_wo_sequence = TabularDataset(test_df)

        assert "sequence" not in tabular_test_wo_sequence.columns
        assert "family_accession" in tabular_test_wo_sequence.columns

        leaderboard = self.tabular_predictor.leaderboard(
            tabular_test_wo_sequence,
            extra_metrics=["log_loss", "pac_score", "accuracy"],
            silent=True,
        )
        performance = self.tabular_predictor.evaluate(
            tabular_test_wo_sequence,
            auxiliary_metrics=True,
            detailed_report=False,
            silent=True,
        )

        return leaderboard, performance

    def _combine_all_embeddings_all_folds(self, dict_of_features):

        all_features_train = self._combine_and_dedupe(dict_of_features["train"])
        all_features_val = self._combine_and_dedupe(dict_of_features["val"])
        all_features_test = self._combine_and_dedupe(dict_of_features["test"])
        return all_features_train, all_features_val, all_features_test

    def _combine_and_dedupe(self, list_of_dfs):
        combined_df = pd.concat(list_of_dfs, axis=1)
        return self._deduplicate_join_columns(combined_df)

    def _deduplicate_join_columns(self, df):
        return df.loc[:, ~df.columns.duplicated()].copy()

    def _load_df(self, path):
        return pd.read_csv(path)

    def convert_df_to_tabular(self, set_of_dfs):
        return list(map(TabularDataset, set_of_dfs))
