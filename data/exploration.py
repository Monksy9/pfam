
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import sequence_length as sl

style.use('seaborn-poster')
style.use('ggplot')
pd.option_context('display.max_rows', None,
                'display.max_columns', None,
                'display.precision', 3,
                )

class DatasetExplorer():
    """
    Top level class responsible for taking in three folds and producing logging/charts on key information.
    """
    
    def __init__(self, train_df, test_df, val_df, top_n_families=10):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.top_n_families = top_n_families        
        self.sequence_length_explorer = sl.SequenceLengthExplorer(self.train_df, self.val_df, self.test_df, top_n_families)
        
    def print_n_rows_df(self):
        
        print("Number of train rows: " + str(self.train_df.shape[0]))
        print("Number of val rows: " + str(self.val_df.shape[0]))
        print("Number of test rows: " + str(self.test_df.shape[0]))
        
    def print_sequence_length_info(self):
        print("Sequence length percentiles: ")
        self.sequence_length_explorer.print_sequence_length()
        print("\n\n")
        
    def plot_major_families(self):
        self.sequence_length_explorer.plot_major_families_by_length()
        
    def plot_major_families_above_threshold(self, length_cutoff):
        self.sequence_length_explorer.plot_major_families_above_n_threshold(length_cutoff=length_cutoff)
        
    def check_for_unseen_families(self):        
        train_families = list(self.train_df['family_accession'].unique())
        test_families = list(self.test_df['family_accession'].unique())
        val_families = list(self.val_df['family_accession'].unique())
        
        new_test_families = self._find_difference_of_families(test_families, train_families)
        new_val_families = self._find_difference_of_families(val_families, train_families)
        
        print("There are " + str(len(new_test_families)) + " new families in test vs train")
        print("There are " + str(len(new_val_families)) + " new families in val vs train \n \n")
    
    def family_summary_text(self):
        no_train_families = self.train_df['family_accession'].nunique()
        no_test_families = self.test_df['family_accession'].nunique()
        no_val_families = self.val_df['family_accession'].nunique()
        
        print("There are " + str(no_train_families) + " unique families in train")
        print("There are " + str(no_test_families) + " unique families in test")
        print("There are " + str(no_val_families) + " unique families in val \n \n")
        
        
        self.check_for_unseen_families()
        
        self.counts_per_family()
        
    def family_summary_plots(self):
        self.plot_countplots()
        
        
    def _find_difference_of_families(self, df_1, df_2):
        return [family for family in df_1 if family not in df_2]
    
    
    def plot_countplots(self, top_n=10):
        
        fig, ax = plt.subplots(1, 1, figsize=(30, 15))
        list_of_top_n_folds = list(map(self._retrieve_top_n_df, (self.train_df, self.test_df, self.val_df))) 
        fold_names = ["train", "test", "val"]
        all_folds_top_n = pd.concat(list_of_top_n_folds, ignore_index=True)
        all_folds_top_n['fold'] = np.repeat(fold_names, list(map(len, list_of_top_n_folds)))
        sns.countplot(x=all_folds_top_n['family_id'], order=all_folds_top_n['family_id'].value_counts().index, hue=all_folds_top_n['fold'])
        title = 'Top ' + str(top_n) + ' families across train, test and val'
        ax.set_title(title)
        
    def _retrieve_top_n_df(self, df):     
        top_n = list(df['family_id'].value_counts().head(self.top_n_families).index)
        return df[df['family_id'].isin(top_n)]
    
    def multiple_records_per_sequence(self):
        counts = self.train_df.groupby('sequence', as_index=False).count() 
        
        more_than_one_sequence = counts[counts['family_accession'] > 1][['sequence', 'sequence_name']].sort_values(by='sequence_name', ascending=False)
        print("There are " + str(more_than_one_sequence['sequence'].nunique()) + " unique sequences, that have multiple family_ids ")
        print("There are " + str(more_than_one_sequence['sequence_name'].sum()) + " rows in total that have multiple family ids \n")

        print("Here's an example:")
    
        most_repeated = more_than_one_sequence['sequence'].values[0]
    
        print(self.train_df[self.train_df['sequence']== most_repeated][['sequence', 'sequence_name']].head(5))
    
        print('\n\n')
    
    def counts_per_family(self):
        
        print("Number of rows per family:")
        print(self.train_df['family_accession'].value_counts().describe())
        print('\n\n')