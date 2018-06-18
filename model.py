import numpy as np
import tensorflow.contrib.keras as K
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


class SVM:
    def __init__(self, **kwargs):
        self.model = svm.LinearSVC(**kwargs)

    def create_model(self, input_shape, hidden_units):
        pass

    def fit(self, x, y, epochs=10, **kwargs):
        self.model.fit(x, y[:, 0])

    def predict(self, x):
        return self.model.predict(x)

    def cross_val(self, X, Y, seed=7):
        model = svm.SVC()
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        scores = cross_val_score(model, X, Y, cv=kfold)
        return scores


class NN:
    def __init__(self):
        pass

    def create_model(self, input_shape=(None, 25), hidden_units=50):
        model = K.models.Sequential()
        model.add(K.layers.Dense(hidden_units, activation='tanh', input_shape=(None, input_shape[1])))
        # model.add(K.layers.Dense(hidden_units, activation='tanh'))
        model.add(K.layers.Dense(2, activation='softmax'))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        self.model = model
        return self.model

    def fit(self, x, y, epochs=10, **kwargs):
        self.model.fit(np.expand_dims(x, axis=1), np.expand_dims(y, axis=1), epochs=epochs, validation_split=0.2, **kwargs)

    def predict(self, x):
        return np.array(self.model.predict(np.expand_dims(x, axis=1)).tolist())

    def cross_val(self, X, Y, seed=7):
        return np.array([0])
        model = KerasClassifier(build_fn=self.create_model, epochs=150, batch_size=10, verbose=0)
        # evaluate using 10-fold cross validation
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        results = cross_val_score(model, X, Y, cv=kfold)
        return results
