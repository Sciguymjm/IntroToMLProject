import pandas as pd

import data
import model
import numpy as np

if __name__ == '__main__':
    n_columns = 25
    df = pd.read_csv('mushrooms.csv')
    for n_columns in range(1, 35):
        data.convert_data_to_dummies('mushrooms.csv', n_columns=n_columns)
        features = pd.read_csv('features.csv', index_col=0)
        labels = pd.read_csv('labels.csv', index_col=0)
        train, test, train_output, test_output = data.train_test_val_split(features, labels)
        #
        # nn = model.NN()
        # nn.create_model(train.shape, 10)
        # nn.fit(np.expand_dims(train.values, axis=1), np.expand_dims(train_output.values, axis=1))
        # output = nn.predict(np.expand_dims(test.values, axis=1))
        # output = np.array(output.tolist())
        # is_edible = output[:, 0, 0] > output[:, 0, 1]
        # test_output  = test_output.values
        # is_edible_test = test_output[:, 0] > test_output[:, 1]
        #
        # acc = np.sum(is_edible == is_edible_test) / test_output.shape[0]
        #
        # print(acc)

        svm = model.SVM()
        svm.fit(train.values, train_output.values[:, 0])
        result = svm.predict(test.values)

        acc = np.sum(result == test_output.values[:, 0]) / len(result)

        print(n_columns, acc, df.columns[n_columns])