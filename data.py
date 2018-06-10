import pandas as pd


def convert_data_to_dummies(fn):
    # TODO: check for null features
    df = pd.read_csv(fn)
    labels = df[df.columns[0]]
    labels.to_csv('labels.csv')
    features = df[df.columns[1:]]
    features_dummies = pd.get_dummies(features)
    features_dummies.to_csv('features.csv')