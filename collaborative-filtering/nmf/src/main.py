from prepare_data import prepare_data
import config
from nmf import nmf
from svd import svd
import argparse

if __name__ == "__main__":


    # loading and preparing data
    X_df = prepare_data().load_data(config.data_filename)
    train_df, test_df = prepare_data().train_test_split(X_df, test_ratio=config.test_size)


    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices={'svd', 'nmf'},
                        default='nmf', help="method. Default is nmf")

    args = parser.parse_args()

    if args.method == 'nmf':
        print nmf().score(train_df, test_df)
    if args.method == 'svd':
        print svd().score(train_df, test_df)
