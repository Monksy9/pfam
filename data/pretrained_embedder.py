from biotransformers.bio_transformers import BioTransformers
import numpy as np
import pandas as pd


class PretrainedEmbedder:
    """
    Class for easily retrieving embeddings from pretrained models available via biotransformers.
    """
    def __init__(self, num_gpus, sequence_length=128, model_name: str = "esm1_t6_43M_UR50S",):
        self.num_gpus = num_gpus
        self.transformer = self.set_transformer(model_name)
        self.sequence_length = sequence_length
        

    def embed_sequences(self, input_df: pd.DataFrame):
        sequences_truncated = input_df["sequence"].apply(
            lambda x: x[0 : self.sequence_length + 1]
        )
        sequences = np.array(sequences_truncated.values)
        esm_embed_arrays = self.transformer.compute_embeddings(sequences)["cls"]
        return self.join_embeddings_to_df(esm_embed_arrays, input_df)

    def set_transformer(self, model_name: str):
        return BioTransformers(model_name, self.num_gpus)

    def join_embeddings_to_df(
        self, esm_embeded_arrays: np.array, input_df: pd.DataFrame
    ):
        embeddings_df = pd.DataFrame(esm_embeded_arrays)
        combined_df = pd.concat(
            [input_df[["sequence", "family_id"]].reset_index(drop=True), embeddings_df],
            axis=1,
        )
        return combined_df
