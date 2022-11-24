import seaborn as sns
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def tsne_plot(df, ax):

    embeddings_only = df.drop(['sequence', 'family_id'], axis=1) 
    X_embedded = TSNE(n_components=2).fit_transform(embeddings_only)
    
    df.loc[:, 'PC1'] = X_embedded[:,0]
    df.loc[:, 'PC2'] = X_embedded[:,1]
    most_freq_classes = list(df['family_id'].value_counts().head(20).index)
    train_df_top10 = df.loc[df.family_id.isin(most_freq_classes)]
    sns.scatterplot(data=train_df_top10, x='PC1', y='PC2', ax=ax, hue='family_id')

def plot_all_embeddings(df, embeddings_to_use, dict_of_features): 
    fig, ax = plt.subplots(len(embeddings_to_use),1, figsize=(20,20))
    for index, embedding in enumerate(embeddings_to_use):
        tsne_plot(df[dict_of_features[embedding]], ax[index])
        ax[i].set_title("Embeddings for " + str(embeddings_to_use))
