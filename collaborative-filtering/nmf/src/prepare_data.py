import pandas as pd
import collections
import numpy as np
from sklearn.model_selection import train_test_split
import config

class prepare_data():
    def __init___(self):
        pass

    def load_data(self, data_filename):
        data_df = pd.read_csv(open(data_filename,'rU'), encoding='utf-8', \
            engine='python', nrows=config.n_rows)
        return data_df

    def train_test_split(self, df, test_ratio):
        train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=42)
        return train_df, test_df
