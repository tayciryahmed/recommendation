from prepare_data import prepare_data
import config
from nmf import nmf

if __name__ == "__main__":

    # loading and preparing data
    X_df = prepare_data().load_data(config.data_filename)
    X_train, test_df = prepare_data().train_test_split(X_df)

    #NMF
    print nmf().score(X_train, test_df)
