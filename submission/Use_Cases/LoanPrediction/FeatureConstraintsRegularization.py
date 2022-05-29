from scipy.optimize import minimize
import numpy as np
import time


class LoanPredictionRegularization:
    def __init__(self, loss_type='sfb'):
        self.beta_init = None
        self.beta = None
        self.X = None
        self.y = None
        self.alpha = None
        self.runtime = None
        self.K = None
        self.loss_type = loss_type

    @staticmethod
    def linear_function(X, W):
        out = W[0]
        for i in range(len(X)):
            out = out + X[i] * W[i + 1]
        return out

    def entropy_loss(self, beta):
        self.beta = beta
        y_hat = self.predict(self.X)[0]
        prob = [1 / (1 + np.exp(-1 * y_hat[i])) for i in range(len(y_hat))]
        if self.loss_type == 'sbr':
            return -sum([np.log(prob[i] + np.exp(-10)) if self.y[i] == 1 else
                         np.log(1 - prob[i] + np.exp(-10)) for i in range(len(self.y))]) / len(self.y) + self.alpha * \
                   sum([prob[i] for i in range(len(self.y)) if
                        self.X['Credit_History'][i] == 0 and self.X['ApplicantIncome'][i] <= self.K[0]])/len(self.y)
        elif self.loss_type == 'sl':
            return -sum([np.log(prob[i] + np.exp(-10)) if self.y[i] == 1 else
                         np.log(1 - prob[i] + np.exp(-10)) for i in range(len(self.y))]) / len(self.y) + self.alpha * \
                   -sum([np.log(1-prob[i]) for i in range(len(self.y)) if
                        self.X['Credit_History'][i] == 0 and self.X['ApplicantIncome'][i] <= self.K[0]])/len(self.y)
        else:
            raise Exception('Please provide a valid loss type: sbr or sl.')

    def learn(self, X, y, alpha=0, K=None):
        self.alpha = alpha
        self.X = X
        self.y = y
        self.K = K
        self.beta_init = [0] * (X.shape[1] + 1)
        print('starting the optimization with alpha:', self.alpha)
        start = time.time()
        res = minimize(self.entropy_loss, self.beta_init, method='BFGS')
        self.runtime = time.time() - start
        self.beta = list(res.x)
        print(self.beta)
        self.beta_init = self.beta

    def predict(self, X):
        y_hat = []
        y_pred = []
        for i in range(X.shape[0]):
            v = LoanPredictionRegularization.linear_function(list(X.iloc[i, :]), self.beta)
            y_hat.append(v)
            y_pred.append(1 if v > 0 else 0)
        return y_hat, y_pred

    def reset(self):
        self.beta_init = None
        self.beta = None
        self.X = None
        self.y = None
        self.alpha = None
