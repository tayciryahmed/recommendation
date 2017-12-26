from scipy.sparse import csc_matrix
import pandas as pd
import collections
import numpy as np

class prepare_data():
    def __init___(self):
        pass

    def load_data(self, data_filename):
        data_df = pd.read_csv(data_filename, nrows=1500)
        return data_df

    def train_test_split(self, df, test_ratio=0.2):
        length = len(df)
        # Generate indices split
        test_indices = np.random.choice(length, int(test_ratio * length), replace=False)
        train_indices = list(set(range(length)).difference(test_indices))

        test_df = df.iloc[test_indices, :]
        train_df = df.iloc[train_indices, :]

        train_sparse_matrix = csc_matrix((train_df['rating'],
            (train_df['new_user'], train_df['new_item']))).toarray()

        return train_sparse_matrix, test_df
