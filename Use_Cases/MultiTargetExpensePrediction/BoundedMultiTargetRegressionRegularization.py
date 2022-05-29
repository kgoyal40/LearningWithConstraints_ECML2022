from scipy.optimize import minimize
import numpy as np
import sys
import pandas as pd
from sklearn.metrics import mean_squared_error


class BoundedPredictionRegularization:
    def __init__(self):
        self.beta_init = None
        self.beta = None
        self.X = None
        self.y = None
        self.alpha = None
        self.K = None

    @staticmethod
    def linear_function(X, W):
        out = W[0]
        for i in range(len(X)):
            out = out + X[i] * W[i + 1]
        return out

    def bounded_prediction_loss(self, beta):
        self.beta = beta
        y_pred = self.predict(self.X, self.y)[0]
        return (sum([sum([(self.y[j][i] - y_pred[j][i]) ** 2 for j in range(len(y_pred))])
                     for i in range(self.X.shape[0])]) + self.alpha * sum(
            [sum([y_pred[j][i] for j in range(len(y_pred))]) -
             (self.X['Total Household Income'][i] - self.K[0]) / self.K[1] if
             sum([y_pred[j][i] for j in range(len(y_pred))]) >
             (self.X['Total Household Income'][i] - self.K[0]) / self.K[1] else 0
             for i in range(self.X.shape[0])] +
            [y_pred[1][i] -
             0.05*(self.X['Total Household Income'][i] - self.K[0]) / self.K[1] if
             y_pred[1][i] >
             0.05*(self.X['Total Household Income'][i] - self.K[0]) / self.K[1] else 0
             for i in range(self.X.shape[0])]
        )) / self.X.shape[0]

    def learn(self, X, y, K=None, alpha=0):
        self.X = X
        self.y = y
        self.K = K
        self.alpha = alpha

        self.beta_init = [1] * (X.shape[1] + 1)
        self.beta_init = self.beta_init * len(y)
        print('starting the optimization with alpha:', alpha)
        res = minimize(self.bounded_prediction_loss, self.beta_init, method='BFGS')
        self.beta = res.x
        print(self.beta)
        self.beta_init = self.beta

    def predict(self, X, y):
        num_targets = int(len(self.beta) / (X.shape[1] + 1))
        weights = [self.beta[i * (X.shape[1] + 1):(i + 1) * (X.shape[1] + 1)] for i in range(num_targets)]
        X_1 = X.copy()
        X_1.insert(loc=0, column='intercept', value=[1] * X.shape[0])
        X_1 = X_1[:].values
        predictions = [list(X_1.dot(pd.Series(w))) for w in weights]
        loss = [mean_squared_error(y[i], predictions[i]) for i in range(len(y))]

        return predictions, loss

    def reset(self):
        self.beta_init = None
        self.beta = None
        self.X = None
        self.y = None
        self.alpha = None
