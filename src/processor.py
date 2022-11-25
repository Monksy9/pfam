import pretrained_embedder as pe
import pct_embedder as pct_e
import os
import load_data as ld
import pandas as pd
import shutil


class Embedder:
    """
    Appies Embedders to amino acid input sequences containing .embed_input_sequences and unload their output locally.
    """

    def __init__(
        self,
        train_size,
        val_size,
        test_size,
        raw_data_path,
        num_gpus: int,
        sequence_length: int = 512,
    ):
        Loader = ld.DataLoader()
        self.train_df, self.val_df, self.test_df = Loader.load_all_three_dfs(
            raw_data_path=raw_data_path,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )
        self.sequence_length = sequence_length
        self.num_gpus = num_gpus

    def embed_input_sequences(self, embedder_name):

        if "esm1" in embedder_name or "bert" in embedder_name:
            print("Using bio-transformers library for:" + str(embedder_name))
            embedder = pe.PretrainedEmbedder(
                model_name=embedder_name,
                sequence_length=self.sequence_length,
                num_gpus=self.num_gpus,
            )
        elif "pct" in embedder_name:
            print("Using percentage embedder")
            embedder = pct_e.SimpleEmbedder()
        else:
            raise Exception("No correct embedder")

        train_embeddings_df = embedder.embed_sequences(self.train_df)
        val_embeddings_df = embedder.embed_sequences(self.val_df)
        test_embeddings_df = embedder.embed_sequences(self.test_df)

        train_embeddings_df = self._append_embedder_name_to_columns(
            train_embeddings_df, embedder_name=embedder_name
        )
        test_embeddings_df = self._append_embedder_name_to_columns(
            test_embeddings_df, embedder_name=embedder_name
        )
        val_embeddings_df = self._append_embedder_name_to_columns(
            val_embeddings_df, embedder_name=embedder_name
        )

        return train_embeddings_df, val_embeddings_df, test_embeddings_df

    def embed_and_unload(self, embedder_name):
        embed_train, embed_test, embed_val = self.embed_input_sequences(embedder_name)
        embedding_path = "embeddings/" + str(embedder_name) + "/"

        if not os.path.isdir(embedding_path):
            os.makedirs(embedding_path)
        else:
            shutil.rmtree(embedding_path)
            os.makedirs(embedding_path)

        embed_train.to_csv(embedding_path + "train.csv", index=False)
        embed_test.to_csv(embedding_path + "test.csv", index=False)
        embed_val.to_csv(embedding_path + "val.csv", index=False)

    def _append_embedder_name_to_columns(
        self, df: pd.DataFrame, embedder_name: str
    ) -> pd.DataFrame:

        df.columns = [
            embedder_name[0:5] + str(col)
            if col not in ["family_accession", "sequence"]
            else col
            for col in df.columns
        ]

        return df


def families_in_train_not_needed(train_df, val_df, test_df):
    train = set(train_df["family_accession"].values)
    families_in_test_val = set(
        list(val_df["family_accession"].values)
        + list(val_df["family_accession"].values)
    )
    families_to_remove = [item for item in train if item not in families_in_test_val]
    print("Number of families removed: " + str(len(families_to_remove)))

    return train_df[~train_df["family_accession"].isin(families_to_remove)]
