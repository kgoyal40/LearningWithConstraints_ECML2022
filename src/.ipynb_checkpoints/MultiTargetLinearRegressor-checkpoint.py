from Optimizer import Optimizer
from z3 import *
from sklearn.metrics import mean_squared_error
import time
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import statistics


class MultiLinearRegressionLearner(Optimizer):
    """
    Basic Linear Regression happens in this class. Inherits the optimizer class where the magic happens
    """

    def __init__(self, verbose=False):
        super().__init__(verbose)
        self.runtime = 0
        self.W_learnt = None
        self.instances_to_change = []
        self.training_losses = []
        self.validation_losses = []
        self.all_weights = []
        self.knowledge_constraint_index = []

    @staticmethod
    def sumproduct(X, W):
        out = W[0]
        for i in range(len(X)):
            out = out + X[i] * W[i + 1]
        return out

    def add_knowledge_constraints(self, X, y, K=None):
        pass

    def add_decision_constraints(self, X, y, alphas=None, learner='maxl'):
        Fs = []
        Fh = []
        if learner == 'maxl':
            for k in range(len(y)):
                y_label = y[k]
                for i in range(X.shape[0]):
                    for j in range(len(alphas[k])):
                        Fh.append(self.T[X.shape[0] * k * len(alphas[k]) + i * len(alphas[k]) + j] ==
                                  And(MultiLinearRegressionLearner.sumproduct(list(X.iloc[i]), self.W[k])
                                      >= y_label[i] - alphas[k][j],
                                      MultiLinearRegressionLearner.sumproduct(list(X.iloc[i]), self.W[k])
                                      <= y_label[i] + alphas[k][j]))

                    Fs.append(Sum([If(self.T[X.shape[0] * k * len(alphas[k]) + i * len(alphas[k]) + j], 1, 0)
                                   for j in range(len(alphas[k]))]) >= int(len(alphas[k])))
        elif learner == 'fumalik':
            for k in range(len(y)):
                y_label = y[k]
                for i in range(X.shape[0]):
                    for j in range(len(alphas[k])):
                        Fh.append(self.T[X.shape[0] * k * len(alphas[k]) + i * len(alphas[k]) + j] ==
                                  And(MultiLinearRegressionLearner.sumproduct(list(X.iloc[i]), self.W[k])
                                      >= y_label[i] - alphas[k][j],
                                      MultiLinearRegressionLearner.sumproduct(list(X.iloc[i]), self.W[k])
                                      <= y_label[i] + alphas[k][j]))

                        Fs.append(self.T[X.shape[0] * k * len(alphas[k]) + i * len(alphas[k]) + j])
        else:
            raise ValueError('Provide a valid learner, either maxl or fumalik')

        self.Soft_Constraints.extend(Fs)
        self.Hard_Constraints.extend(Fh)

    def add_weight_constraints(self, low=-1000, high=1000):
        Fh = []
        for W in self.W:
            for w in W:
                Fh.append(And(w >= low, w <= high))
        self.Hard_Constraints.extend(Fh)

    def learn(self, X, y, M=None, relaxation=1, K=None, weight_limit=100,
              validation_set=None, batch_size=50, epochs=1, learning_rate=0.1,
              early_stopping=True, learner='maxl', step_size=1, waiting_period=5):

        self.solver.set('timeout', waiting_period * 1000)
        early_stopping = early_stopping
        if M is None:
            print('please provide a list of margins')
            sys.exit(0)

        # for classification problems, bigger margin is stricter
        M.sort(reverse=True)
        alphas = [[max(y[i]) * m for m in M] for i in range(len(y))]

        # relaxation variable used in maxsat
        self.relaxation = relaxation

        # parameter initialization
        self.W = [[Real('w_{}_{}'.format(i, j)) for i in range((X.shape[1] + 1))] for j in range(len(y))]

        all_indices = list(range(X.shape[0]))
        all_batches = [all_indices[i * batch_size: (i + 1) * batch_size]
                       for i in range(int(len(all_indices) / batch_size))]
        current_weights = []
        gradient_direction = []
        variable_learning_rate = learning_rate

        stop_learning = False
        for ep in range(epochs):
            for j in range(len(all_batches)):
                active_indices = all_batches[j]
                print('number of instances being considered:', len(active_indices))
                if K is not None:
                    if len(self.training_losses) % step_size == 0:
                        self.add_knowledge_constraints(X, y, K)
                        print('knowledge constraints added')
                        variable_learning_rate = learning_rate
                        print('knowledge constraints added, maximal step size is {}'.format(variable_learning_rate))
                    else:
                        variable_learning_rate = 0.05
                        print('knowledge constraints not added, maximal step size is {}'.format(variable_learning_rate))

                self.add_weight_constraints(low=-weight_limit, high=weight_limit)

                if len(current_weights) > 0:
                    print(gradient_direction)
                    self.Hard_Constraints.append(
                        And([And([If(current_weights[i][k] == -weight_limit, True,
                                     And(self.W[i][k] < current_weights[i][k], self.W[i][k] >= current_weights[i][k] -
                                         variable_learning_rate)) if gradient_direction[i][k] == 1
                                  else If(current_weights[i][k] == weight_limit, True,
                            And(self.W[i][k] > current_weights[i][k], self.W[i][k] <= current_weights[i][k] +
                                variable_learning_rate))
                                  for k in range(len(self.W[i]))]) for i in range(len(self.W))]))

                self.T = BoolVector('t', X.shape[0] * len(M) * len(y))
                X_sub = X.iloc[active_indices, :]
                y_sub = [[y[i][j] for j in active_indices] for i in range(len(y))]

                X_sub.reset_index(drop=True, inplace=True)
                self.add_decision_constraints(X_sub, y_sub, alphas=alphas, learner=learner)

                # optimization step (maxsat)
                start = time.time()
                if learner == 'maxl':
                    self.OptimizationMaxL(length=len(M))
                elif learner == 'fumalik':
                    self.OptimizationFuMalik(length=len(M))
                else:
                    raise ValueError('Provide a valid learner, either maxl or fumalik')
                self.runtime = self.runtime + time.time() - start

                if self.out == sat:
                    self.W_learnt = [
                        [self.solver.model()[u].numerator_as_long() / self.solver.model()[u].denominator_as_long()
                         for u in w] for w in self.W]
                    print('learned parameters')
                    print(self.W_learnt)
                    self.all_weights.append(self.W_learnt)
                    gradient_all = self.calculate_gradient(X, y, self.W_learnt)
                    gradient_values = gradient_all[0]
                    gradient_direction = [[1 if g >= 0 else -1 for g in G] for G in gradient_values]
                    print('gradient:', gradient_values)
                    current_weights = self.W_learnt.copy()
                    print('runtime:', self.runtime)
                    training_prediction = self.predict(X, y)
                    print('training mse:', training_prediction[1])
                    if validation_set is not None:
                        self.validation_losses.append(self.calculate_gradient(validation_set[0], validation_set[1],
                                                                              self.W_learnt)[1])
                    self.training_losses.append(gradient_all[1])
                    if variable_learning_rate == 0.05:
                        self.knowledge_constraint_index.append(0)
                    else:
                        self.knowledge_constraint_index.append(1)
                else:
                    print('found unknown. randomizing gradients')
                    gradient_direction = [[g * -1 for g in G] for G in gradient_direction]

                self.Soft_Constraints = []
                self.Hard_Constraints = []
                self.T = None
                self.W_learnt = None
                self.out = None
                self.solver.reset()

                if early_stopping:
                    if len(self.training_losses) >= 400 and len(self.training_losses) % 100 == 0:
                        if validation_set is not None:
                            avg_loss_last_50 = statistics.mean(self.validation_losses[-200:])
                            avg_loss_last_100_50 = statistics.mean(self.validation_losses[-400:-200])
                            if (avg_loss_last_100_50 - avg_loss_last_50) / avg_loss_last_100_50 <= 0.02:
                                print('stopping the iterations as there is no improvement')
                                stop_learning = True
                                break
                        else:
                            avg_loss_last_50 = statistics.mean(self.training_losses[-100:])
                            avg_loss_last_100_50 = statistics.mean(self.training_losses[-200:-100])
                            if (avg_loss_last_100_50 - avg_loss_last_50) / avg_loss_last_100_50 <= 0.02:
                                print('stopping the iterations as there is no improvement')
                                stop_learning = True
                                break

                # if len(self.training_losses) == 15:
                #     stop_learning = True
                #     break

            if stop_learning:
                break

        self.W_learnt = current_weights
        print('all training mses:', self.training_losses)
        print('all validation mses:', self.validation_losses)

        if validation_set is not None:
            self.W_learnt = self.all_weights[self.validation_losses.index(min(self.validation_losses))]
        else:
            self.W_learnt = self.all_weights[self.training_losses.index(min(self.training_losses))]

        self.instances_to_change = []

    def get_plot(self, plot_suffix=''):
        plt.figure()
        plt.plot(self.training_losses)
        plt.xlabel('batches')
        plt.ylabel('training mean squared error')

        if K is None:
            filename = 'training_mse_without_knowledge_' + plot_suffix + '.png'
        else:
            filename = 'training_mse_with_knowledge_' + plot_suffix + '.png'
        plt.savefig(filename)

        if len(self.validation_losses) > 0:
            plt.figure()
            plt.plot(self.validation_losses)
            plt.xlabel('batches')
            plt.ylabel('mean squared error')

            if K is None:
                filename = 'validation_mse_without_knowledge_' + plot_suffix + '.png'
            else:
                filename = 'validation_mse_with_knowledge_' + plot_suffix + '.png'

            plt.savefig(filename)

    def calculate_gradient(self, X, y, weights):
        gradients = []
        pred = self.predict(X, y, weights=weights)
        predictions = pred[0]
        loss = pred[1]

        X_1 = X.copy()
        X_1.insert(loc=0, column='intercept', value=[1] * X.shape[0])
        X_1 = X_1[:].values

        for k in range(len(y)):
            residual = (-2 / X.shape[0]) * (y[k] - predictions[k])
            g = list(pd.Series(residual).dot(X_1))
            gradients.append(g)

        return gradients, loss

    def predict(self, X, y, weights=None):
        if weights is None:
            weights = self.W_learnt

        X_1 = X.copy()
        X_1.insert(loc=0, column='intercept', value=[1] * X.shape[0])
        X_1 = X_1[:].values

        predictions = [X_1.dot(pd.Series(w)) for w in weights]
        loss = [mean_squared_error(y[i], predictions[i]) for i in range(len(y))]

        return predictions, loss
