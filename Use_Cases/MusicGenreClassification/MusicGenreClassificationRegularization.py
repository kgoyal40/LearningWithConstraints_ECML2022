from scipy.optimize import minimize
import numpy as np
import sys
import pandas as pd
from sklearn.metrics import mean_squared_error


class MusicGenreClassificationRegularization:
    def __init__(self, loss_type='sbr'):
        self.beta_init = None
        self.beta = None
        self.X = None
        self.y = None
        self.alpha = None
        self.targets = None
        self.loss_type = loss_type

    @staticmethod
    def linear_function(X, W):
        out = W[0]
        for i in range(len(X)):
            out = out + X[i] * W[i + 1]
        return out

    def multiclass_entropy_loss(self, beta):
        self.beta = beta
        y_hat = self.predict(self.X)[0]
        prob = [[1 / (1 + np.exp(-1 * y_hat[i][j])) for j in range(len(y_hat[i]))]
                for i in range(len(self.targets))]
        if self.loss_type == 'sbr':
            return -sum(
                [sum([np.log(prob[j][i] + np.exp(-10)) if self.y[j][i] == 1 else np.log(1 - prob[j][i] + np.exp(-10))
                      for j in range(len(self.targets))]) for i in range(self.X.shape[0])]) / self.X.shape[0] + \
                   self.alpha * (1 - sum([max(prob[3][i], prob[4][i]) for i in range(self.X.shape[0]) if
                                          self.X['group_The Beatles'][i] == 1]) / self.X.shape[0])
        elif self.loss_type == 'sl':
            return -sum(
                [sum([np.log(prob[j][i] + np.exp(-10)) if self.y[j][i] == 1 else np.log(1 - prob[j][i] + np.exp(-10))
                      for j in range(len(self.targets))]) for i in range(self.X.shape[0])]) / self.X.shape[0] + \
                   self.alpha * -sum([np.log((prob[3][i] + prob[4][i]) *
                                             (1 - prob[0][i]) * (1 - prob[1][i]) * (1 - prob[2][i]))
                                      for i in range(self.X.shape[0])
                                      if self.X['group_The Beatles'][i] == 1]) / self.X.shape[0]
        else:
            raise Exception('Please provide a valid loss type: sbr or sl.')

    def learn(self, X, y, alpha=0):
        self.X = X
        self.targets = np.unique(y)
        print(self.targets)
        # ['Classical' 'Electronic' 'Metal' 'Pop' 'Rock']
        self.y = [[1 if x == self.targets[i] else 0 for x in y] for i in range(len(self.targets))]
        self.alpha = alpha

        self.beta_init = [1] * (X.shape[1] + 1)
        self.beta_init = self.beta_init * len(self.targets)
        print('starting the optimization with alpha:', alpha)
        res = minimize(self.multiclass_entropy_loss, self.beta_init, method='BFGS')
        self.beta = res.x
        print(self.beta)
        self.beta_init = self.beta

    def predict(self, X):
        num_targets = int(len(self.beta) / (X.shape[1] + 1))
        weights = [self.beta[i * (X.shape[1] + 1):(i + 1) * (X.shape[1] + 1)] for i in range(num_targets)]

        X_1 = X.copy()
        X_1.insert(loc=0, column='intercept', value=[1] * X.shape[0])
        X_1 = X_1[:].values

        y_hat = [list(X_1.dot(pd.Series(weights[i]))) for i in range(len(self.targets))]
        y_pred = [[y_hat[j][i] for j in range(len(self.targets))] for i in range(X.shape[0])]
        y_pred = [self.targets[p.index(max(p))] for p in y_pred]
        return y_hat, y_pred

    def reset(self):
        self.beta_init = None
        self.beta = None
        self.X = None
        self.y = None
        self.alpha = None
        self.targets = None
