import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
import pandas as pd

pd.set_option("mode.chained_assignment", None)
warnings.simplefilter(action="ignore", category=FutureWarning)  # TSNE future warnings


def tsne_plot(df: pd.DataFrame, ax):
    embeddings_only = df.drop(["sequence", "family_accession"], axis=1)
    tsne_array = TSNE(n_components=2, n_jobs=-1, random_state=1).fit_transform(
        embeddings_only
    )

    df.loc[:, "PC1"] = pd.Series(tsne_array[:, 0]).copy(deep=False)
    df.loc[:, "PC2"] = pd.Series(tsne_array[:, 1]).copy(deep=False)
    most_freq_classes = list(df["family_accession"].value_counts().head(20).index)
    train_df_top10 = df.loc[df.family_accession.isin(most_freq_classes)].copy()
    sns.scatterplot(
        data=train_df_top10, x="PC1", y="PC2", ax=ax, hue="family_accession"
    )


def plot_all_embeddings(
    df: pd.DataFrame, embeddings_to_use: list, dict_of_features: dict
):
    """Given a list of embeddings that have been produced, create t-sne plots

    Args:
        df: dataframe containing all embeddings
        embeddings_to_use: _description_
        dict_of_features: _description_
    """
    fig, ax = plt.subplots(len(embeddings_to_use), 1, figsize=(20, 20))
    for index, embedding in enumerate(embeddings_to_use):
        tsne_plot(df[dict_of_features[embedding]], ax[index])
        ax[index].set_title("Embeddings for " + str(embedding))
