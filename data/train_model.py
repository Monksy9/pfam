import pandas as pd

class TrainModel():
    
    def __init__(self, embeddings_to_use):
        
        self.embeddings_to_use = embeddings_to_use
        
    def load_data(self):
        
        train_dfs = []
        val_dfs = []
        test_dfs = []
        dict_of_columns = {}
        for embedder in self.embeddings_to_use:
            path = 'embeddings/' + str(embedder) + '/'
            train_df = pd.read_csv(path + 'train.csv')
            train_dfs.append(train_df)
            test_dfs.append(pd.read_csv(path + 'test.csv'))
            val_dfs.append(pd.read_csv(path + 'val.csv'))
            dict_of_columns[embedder] = train_df.columns 
            
        all_features_train = pd.concat(train_dfs, axis=1)
        all_features_test = pd.concat(test_dfs, axis=1)
        all_features_val = pd.concat(val_dfs, axis=1) 
        all_features_train = all_features_train.loc[:,~all_features_train.columns.duplicated()].copy()
        all_features_test = all_features_test.loc[:,~all_features_test.columns.duplicated()].copy()
        all_features_val = all_features_val.loc[:,~all_features_val.columns.duplicated()].copy()
        
        return dict_of_columns, all_features_train, all_features_val, all_features_test