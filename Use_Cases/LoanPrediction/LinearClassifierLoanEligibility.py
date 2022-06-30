import sys

sys.path.insert(0, 'path to src folder')
from LinearClassifier import *
import time
import torch


class LinearClassificationLearnerLoan(LinearClassificationLearner):

    def __init__(self, verbose=False):
        super().__init__(verbose)
        self.income = 0
        self.alpha = 0

    def add_knowledge_constraints(self, X, y, K=None):
        if K[1] is not None:
            self.income = K[0]
            self.alpha = K[1]
            print('adding knowledge constraint')
            _X = RealVector('_x', X.shape[1])
            self.Hard_Constraints.append(ForAll(_X, Implies(And([_X[4] == 0] +
                                                                [_X[0] <= self.income] +
                                                                [And(_X[i] <= 1,
                                                                     _X[i] >= 0)
                                                                 for i in range(X.shape[1])]),
                                                            LinearClassificationLearnerLoan.sumproduct(_X,
                                                                                                       self.W) < 0)))

    def predict(self, X, y, weights=None):
        if weights is None:
            weights = self.W_learnt

        X_1 = X.copy()
        X_1.insert(loc=0, column='intercept', value=[1] * X.shape[0])
        X_1 = X_1[:].values
        confidence = X_1.dot(pd.Series(weights))
        predictions = [1 if c > 0 else 0 for c in confidence]

        predictions_correct = [1 if y[i] == predictions[i] else 0 for i in range(len(predictions))]

        return predictions, sum(predictions_correct), sum(predictions_correct) / len(predictions_correct), confidence

    def calculate_gradient(self, X, y, weights):
        weight_tensors = torch.tensor(weights, requires_grad=True)

        X_1 = X.copy()
        X_1.insert(loc=0, column='intercept', value=[1] * X.shape[0])
        X_1 = X_1[:].values
        X_1 = torch.tensor(X_1)
        h = torch.matmul(X_1, weight_tensors.double())

        Z = [1 / (1 + torch.exp(-h[i])) for i in range(len(y))]
        loss = -sum([torch.log(Z[i] + torch.exp(torch.tensor([-15.0]))) if y[i] == 1 else
                     torch.log(1 - Z[i] + torch.exp(torch.tensor([-15.0])))
                     for i in range(len(y))]) / len(y)
        loss.backward()
        gradients = weight_tensors.grad
        gradients = gradients.detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()

        return gradients, loss.item()

    def calculate_gradient_non_regularized(self, X, y, weights):
        gradient = []
        h = self.predict(X, y, weights=weights)[3]
        Z = [1 / (1 + np.exp(-h[i])) for i in range(len(y))]
        loss = log_loss(y, Z)
        Z_Y = [Z[i] - y[i] for i in range(len(y))]
        gradient.append(sum(Z_Y))
        gradient.extend(list(X.T.dot(Z_Y)))
        gradient = [g / X.shape[0] for g in gradient]
        print(gradient)
        return gradient, loss
