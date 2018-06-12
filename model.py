import tensorflow.contrib.keras as K
from sklearn import svm


class SVM:
    def __init__(self):
        self.model = svm.SVC()

    def create_model(self, input_shape, hidden_units):
        pass

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)


class NN:
    def __init__(self):
        pass

    def create_model(self, input_shape, hidden_units):
        model = K.models.Sequential()
        model.add(K.layers.Dense(hidden_units, activation='tanh', input_shape=(None, input_shape[1])))
        # model.add(K.layers.Dense(hidden_units, activation='tanh'))
        model.add(K.layers.Dense(2, activation='softmax'))
        model.compile(optimizer='adam', loss='mse')
        self.model = model

    def fit(self, x, y):
        self.model.fit(x, y, epochs=10)

    def predict(self, x):
        return self.model.predict(x)
