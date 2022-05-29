import sys

sys.path.insert(0, '/kshitijgoyal/desktop/LearningWithConstraints_ECML_2/src')
from LinearClassifierMultiClass import *
import torch


class MusicLinearClassifier(LinearClassificationMultiClassLearner):

    def __init__(self, verbose=False):
        super().__init__(verbose)
        self.alpha = 0

    def add_knowledge_constraints(self, X, y, K=None):
        if K is not None:
            self.alpha = K  # value of k is the value of regularization parameter
            print('Adding hierarchy constraint')
            beatles_index = list(X.columns).index('group_The Beatles')
            X1 = RealVector('x1', X.shape[1])
            print(np.unique(y))
            self.Hard_Constraints.append(ForAll(X1, Implies(And([And(X1[i] <= max(list(X.iloc[:, i])),
                                                                     X1[i] >= min(list(X.iloc[:, i])))
                                                                 for i in range(X.shape[1])] +
                                                                [X1[beatles_index] == 1]),
                                                            And([
                                                                LinearClassificationMultiClassLearner.sumproduct(
                                                                    X1, self.W[0]) < 0,
                                                                LinearClassificationMultiClassLearner.sumproduct(
                                                                    X1, self.W[1]) < 0,
                                                                LinearClassificationMultiClassLearner.sumproduct(
                                                                    X1, self.W[2]) < 0,
                                                                Or(LinearClassificationMultiClassLearner.sumproduct(
                                                                    X1, self.W[3]) > 0,
                                                                   LinearClassificationMultiClassLearner.sumproduct(
                                                                       X1, self.W[4]) > 0
                                                                   )
                                                            ]))))

    def predict(self, X, y):
        weights = self.W_learnt
        X_1 = X.copy()
        X_1.insert(loc=0, column='intercept', value=[1] * X.shape[0])
        X_1 = X_1[:].values

        confidence = [list(X_1.dot(pd.Series(weights[i]))) for i in range(len(self.targets))]
        predictions = [[confidence[j][i] for j in range(len(self.targets))] for i in range(X.shape[0])]
        predictions = [self.targets[p.index(max(p))] for p in predictions]
        predictions_correct = [1 if y[i] == predictions[i] else 0 for i in range(len(predictions))]

        return predictions, sum(predictions_correct), sum(predictions_correct) / len(predictions_correct), confidence

    def calculate_gradient(self, X, y):
        weight_tensors = torch.tensor(self.W_learnt, requires_grad=True)

        X_1 = X.copy()
        X_1.insert(loc=0, column='intercept', value=[1] * X.shape[0])
        X_1 = X_1[:].values
        X_1 = torch.tensor(X_1)
        H = [torch.matmul(X_1, w.double()) for w in weight_tensors]

        y_label = [[1 if x == self.targets[i] else 0 for x in y] for i in range(len(self.targets))]
        Z = [[1 / (1 + torch.exp(-1 * H[i][j])) for j in range(len(H[i]))]
             for i in range(len(self.targets))]

        loss = -sum(
            [sum([torch.log(Z[j][i] + torch.exp(torch.tensor([-10.0]))) if y_label[j][i] == 1
                  else torch.log(1 - Z[j][i] + torch.exp(torch.tensor([-10.0])))
                  for j in range(len(self.targets))])
             for i in range(X.shape[0])]) / X.shape[0] + self.alpha * (
                       1 - sum([torch.max(Z[3][i], Z[4][i]) for i in range(X.shape[0]) if
                                X['group_The Beatles'][i] == 1]) / X[X['group_The Beatles'] == 1].shape[0])

        loss.backward()
        gradients = weight_tensors.grad
        gradients = gradients.detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()
        return gradients, loss.item()
