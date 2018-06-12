import numpy as np
import pandas as pd


def convert_data_to_dummies(fn, n_columns=None):
    # TODO: check for null features
    df = pd.read_csv(fn)
    labels = df[df.columns[0]]
    labels = pd.get_dummies(labels)
    labels.to_csv('labels.csv')
    features = df[df.columns[1:]]
    if n_columns is not None:
        features = features[features.columns[:n_columns]]
    features_dummies = pd.get_dummies(features)
    features_dummies.to_csv('features.csv')


def train_test_val_split(input_df: pd.DataFrame, output_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    shape = input_df.shape
    r = np.random.rand(shape[0])
    train_selector = r < 0.8
    train = input_df[train_selector]
    test = input_df[~train_selector]
    train_output = output_df[train_selector]
    test_output = output_df[~train_selector]
    return train, test, train_output, test_output
