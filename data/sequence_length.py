import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.float_format', lambda x: '%.3f' % x)

class SequenceLengthExplorer():
    
    def __init__(self, train_df, test_df, val_df, top_n_families):
        self.train_df = self._create_sequence_length_column(train_df)
        self.val_df = self._create_sequence_length_column(val_df)
        self.test_df = self._create_sequence_length_column(test_df)
        self.top_n_families = top_n_families
        
    def print_sequence_length(self):
        self.print_length_percentiles()
        self.plot_sequence_length_by_fold()    
        
    def _create_sequence_length_column(self, df):
        df['sequence_length'] = df['sequence'].apply(lambda x: len(x))
        return df
    
    def print_length_percentiles(self):
        print(self.train_df['sequence_length'].describe(percentiles=[0.05, 0.1, 0.9, 0.95]))
        
    def plot_sequence_length_by_fold(self):
        fig, ax = plt.subplots(1,3, figsize=(30,10))
        sns.histplot(self.train_df['sequence_length'], ax=ax[0], kde=True, color='b').set_title('Sequence length train')
        sns.histplot(self.test_df['sequence_length'], ax=ax[1], kde=True, color='b').set_title('Sequence length test')
        sns.histplot(self.val_df['sequence_length'], ax=ax[2], kde=True, color='b').set_title('Sequence length val')

        ax[0].axvline(x=512, color='r')
        ax[1].axvline(x=512, color='r')
        ax[2].axvline(x=512, color='r')
        
    def plot_major_families_by_length(self):
        top_n_families = list(self.train_df['family_id'].value_counts().head(self.top_n_families).index)
        top_n_df = self.train_df[self.train_df['family_id'].isin(top_n_families)]
        fig, ax = plt.subplots(1,1, figsize=(30,10))
        sns.histplot(data=top_n_df, x='sequence_length', hue='family_id', ax=ax, multiple='stack')
        
        
    def plot_major_families_above_n_threshold(self, length_cutoff):
        long_sequences = self.train_df[self.train_df['sequence_length']>length_cutoff]
        top_n_families = list(long_sequences['family_id'].value_counts().head(self.top_n_families).index)
        top_n_df = long_sequences[long_sequences['family_id'].isin(top_n_families)]
        fig, ax = plt.subplots(1,1, figsize=(30,10))
        sns.histplot(data=top_n_df, x='sequence_length', hue='family_id', ax=ax, multiple='stack')
        