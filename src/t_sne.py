import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)  # TSNE future warnings


def tsne_plot(df, ax):

    embeddings_only = df.drop(["sequence", "family_accession"], axis=1)
    tsne_array = TSNE(n_components=2, n_jobs=-1, random_state=1).fit_transform(
        embeddings_only
    )

    df.loc[:, "PC1"] = tsne_array[:, 0]
    df.loc[:, "PC2"] = tsne_array[:, 1]
    most_freq_classes = list(df["family_accession"].value_counts().head(20).index)
    train_df_top10 = df.loc[df.family_accession.isin(most_freq_classes)]
    sns.scatterplot(
        data=train_df_top10, x="PC1", y="PC2", ax=ax, hue="family_accession"
    )


def plot_all_embeddings(df, embeddings_to_use, dict_of_features):
    fig, ax = plt.subplots(len(embeddings_to_use), 1, figsize=(20, 20))
    for index, embedding in enumerate(embeddings_to_use):
        tsne_plot(df[dict_of_features[embedding]], ax[index])
        ax[index].set_title("Embeddings for " + str(embedding))
