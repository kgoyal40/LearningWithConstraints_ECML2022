import sys

sys.path.insert(0, '/home/kshitij/LearningWithConstraints_ECML_2/src')
from MultiTargetLinearRegressor import *
from z3 import *
import torch


class BoundedMultiTargetLearner(MultiLinearRegressionLearner):

    def __init__(self, verbose=False):
        super().__init__(verbose)
        self.K = None
        self.alpha = 10000

    def add_knowledge_constraints(self, X, y, K=None):
        if K is not None:
            self.K = K[:2]
            self.alpha = K[2]
            print('Adding Knowledge Constraints')
            _X = RealVector('x', X.shape[1])
            self.Hard_Constraints.append(ForAll(_X, Implies(And([And(_X[i] <= (1 + 0.1),
                                                                     _X[i] >= (0 - 0.1))
                                                                 for i in range(X.shape[1])]),
                                                            And(Sum(
                                                                [MultiLinearRegressionLearner.sumproduct(_X, self.W[j])
                                                                 for j in range(len(y))]) <= 2 * (_X[-1] - K[0]) / K[1],
                                                                MultiLinearRegressionLearner.sumproduct(_X,
                                                                                                        self.W[1]) <=
                                                                0.05 * (_X[-1] - K[0]) / K[1])
                                                            )))

    def calculate_gradient(self, X, y, weights):
        weight_tensors = torch.tensor(weights, requires_grad=True)

        X_1 = X.copy()
        X_1.insert(loc=0, column='intercept', value=[1] * X.shape[0])
        X_1 = X_1[:].values
        X_1 = torch.tensor(X_1)
        pred = [torch.matmul(X_1, w.double()) for w in weight_tensors]

        loss = (sum([sum([(y[j][i] - pred[j][i]) ** 2 for j in range(len(pred))])
                     for i in range(X.shape[0])]) + self.alpha * (sum(
            [sum([pred[j][i] for j in range(len(pred))]) -
             2 * (X['Total Household Income'][i] - self.K[0]) / self.K[1] if
             sum([pred[j][i] for j in range(len(pred))]) >
             2 * (X['Total Household Income'][i] - self.K[0]) / self.K[1] else 0
             for i in range(X.shape[0])]) + sum([pred[1][i] -
                                                 0.05 * (X['Total Household Income'][i] - self.K[0]) / self.K[1] if
                                                 pred[1][i] >
                                                 0.05 * (X['Total Household Income'][i] - self.K[0]) / self.K[1] else 0
                                                 for i in range(X.shape[0])]))) / X.shape[0]

        loss.backward()
        gradients = weight_tensors.grad
        gradients = gradients.detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()
        return gradients, loss.item()

    def predict(self, X, y, weights=None):
        if weights is None:
            weights = self.W_learnt

        X_1 = X.copy()
        X_1.insert(loc=0, column='intercept', value=[1] * X.shape[0])
        X_1 = X_1[:].values

        predictions = [X_1.dot(pd.Series(w)) for w in weights]
        loss = [mean_squared_error(y[i], predictions[i]) for i in range(len(y))]

        return predictions, loss
