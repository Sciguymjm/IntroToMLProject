import matplotlib.pyplot as plt
import sklearn
from matplotlib import rcParams
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

rcParams.update({'figure.autolayout': True})
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    sns.set()
except:
    print('not using seaborn')
import data
import model
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)
    plt.show()



def plot_coefficients(classifier, feature_names, top_features=10):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure()
    plt.gcf().subplots_adjust(bottom=0.2)
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(0.5 + np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.title('Feature Importance in Classifying as Edible')
    plt.ylabel('Importance')
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()


def test_n_columns(df, low, high):
    for n_columns in range(low, high):
        data.convert_data_to_dummies('mushrooms.csv', n_columns=n_columns)
        features = pd.read_csv('features.csv', index_col=0)
        labels = pd.read_csv('labels.csv', index_col=0)
        train, test, train_output, test_output = data.train_test_val_split(features, labels)
        #

        # print(acc)

        svm = model.SVM()
        svm.fit(train.values, train_output.values[:, 0])
        result = svm.predict(test.values)

        acc = np.sum(result == test_output.values[:, 0]) / len(result)

        print(n_columns, acc, df.columns[n_columns])
        print()


def test_network(train, test, train_output, test_output, model_type, plot_features=False, neurons=10):

    md = model_type()
    md.create_model(train.shape, hidden_units=neurons)
    md.fit(train.values, train_output.values, epochs=1)
    result = md.predict(test.values)
    if plot_features:
        matrix = sklearn.metrics.confusion_matrix(test_output.values[:, 0], result)
        acc = np.sum(result == test_output.values[:, 0]) / len(result)
    else:
        is_edible = result[:, 0, 0] > result[:, 0, 1]
        test_output  = test_output.values
        is_edible_test = test_output[:, 0] > test_output[:, 1]
        matrix = sklearn.metrics.confusion_matrix(is_edible_test, is_edible)

        acc = np.sum(is_edible == is_edible_test) / test_output.shape[0]

    print(acc)
    if plot_features:
        plot_coefficients(md.model, list(features.columns))
        print(md.cross_val(train, train_output.values[:, 0]).mean())

    return acc, matrix


if __name__ == '__main__':
    data.convert_data_to_dummies('mushrooms.csv')
    features = pd.read_csv('features.csv', index_col=0)
    labels = pd.read_csv('labels.csv', index_col=0)
    train, test, train_output, test_output = data.train_test_val_split(features, labels)
    n_columns = 25
    # test_n_columns(df, 1, 35)
    # accs = {k: [] for k in range(1, 11)}
    # for k in range(10):
    #     for n in range(1, 11):
    #         v = test_network(train, test, train_output, test_output, model.NN, neurons=n)
    #         accs[n].append(v[0])
    #
    # accs_mean = np.array([np.array(k).mean() for k in accs.values()])
    # plt.xlabel('Number of neurons in hidden layer')
    # plt.ylabel('Accuracy (%)')
    # plt.plot(list(accs.keys()), accs_mean * 100.0)
    # plt.show()
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    scores = ['precision', 'recall']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(train, train_output.values[:, 0])

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = test_output.values[:, 0], clf.predict(test)
        plot_confusion_matrix(sklearn.metrics.confusion_matrix(y_true, y_pred), ['Edible', 'Poisonous'])
        print(classification_report(y_true, y_pred))
        print()
    # v = (test_network(train, test, train_output, test_output, model.NN, False))
    # plot_confusion_matrix(v[1], ['Edible', 'Poisonous'])