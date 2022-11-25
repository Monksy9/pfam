from biotransformers.bio_transformers import BioTransformers
import numpy as np
import pandas as pd


class PretrainedEmbedder:
    """
    Class for easily retrieving embeddings from pretrained models available via biotransformers.
    """

    def __init__(
        self,
        num_gpus,
        sequence_length: int,
        model_name: str,
    ):
        self.num_gpus = num_gpus
        self.transformer = self.set_transformer(model_name)
        self.sequence_length = sequence_length

    def embed_sequences(self, input_df: pd.DataFrame):
        """
        Given an input dataframe, with the column "sequence" this will created the embeddings for a pretrained embedder from biotransformers

        Args:
            input_df (pd.DataFrame): input dataframe containing sequence column.

        Returns:
            _type_: dataframe containing embeddings, target and sequence.
        """
        sequences_truncated = input_df["sequence"].apply(
            lambda x: x[0 : self.sequence_length + 1]
        )
        sequences = np.array(sequences_truncated.values)
        esm_embed_arrays = self.transformer.compute_embeddings(sequences)["mean"]
        return self.join_embeddings_to_df(esm_embed_arrays, input_df)

    def set_transformer(self, model_name: str):
        return BioTransformers(model_name, self.num_gpus)

    def join_embeddings_to_df(
        self, esm_embeded_arrays: np.array, input_df: pd.DataFrame
    ):
        embeddings_df = pd.DataFrame(esm_embeded_arrays)
        combined_df = pd.concat(
            [
                input_df[["sequence", "family_accession"]].reset_index(drop=True),
                embeddings_df,
            ],
            axis=1,
        )
        return combined_df
