from Optimizer import Optimizer
from z3 import *
import time
import random
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
import statistics


class LinearClassificationMultiClassLearner(Optimizer):
    """
    MultiClass Linear Classification happens in this class.
    Inherits the optimizer class where the magic happens
    """

    def __init__(self, verbose=False):
        super().__init__(verbose)
        self.runtime = 0
        self.W_learnt = None
        self.highest_degree = None
        self.instances_to_change = []
        self.training_accuracies = []
        self.validation_accuracies = []
        self.training_losses = []
        self.validation_losses = []
        self.all_weights = []
        self.targets = []
        self.knowledge_constraint_index = []

    @staticmethod
    def sumproduct(X, W):
        out = W[0]
        for i in range(len(X)):
            out = out + X[i] * W[i + 1]
        return out

    def add_knowledge_constraints(self, X, y, K=None):
        pass

    def add_decision_constraints(self, X, y, M=None, learner='maxl'):
        Fs = []
        Fh = []

        if learner == 'maxl':
            for i in range(len(self.targets)):
                print('i =', i)
                y_label = [1 if x == self.targets[i] else 0 for x in y]
                print(self.W[i])
                for j in range(X.shape[0]):
                    if y_label[j] == 1:
                        for k in range(len(M)):
                            Fh.append(
                                self.T[X.shape[0] * i * len(M) + j * len(M) + k] ==
                                (LinearClassificationMultiClassLearner.sumproduct(list(X.iloc[j]), self.W[i]) > M[k]))

                        Fs.append(Sum([If(self.T[X.shape[0] * i * len(M) + j * len(M) + k], 1, 0)
                                       for k in range(len(M))]) >= len(M))
                    else:
                        for k in range(len(M)):
                            Fh.append(
                                self.T[X.shape[0] * i * len(M) + j * len(M) + k] ==
                                (LinearClassificationMultiClassLearner.sumproduct(list(X.iloc[j]), self.W[i]) < -M[k]))

                        Fs.append(Sum([If(self.T[X.shape[0] * i * len(M) + j * len(M) + k], 1, 0)
                                       for k in range(len(M))]) >= len(M))
        elif learner == 'fumalik':
            for i in range(len(self.targets)):
                print('i =', i)
                y_label = [1 if x == self.targets[i] else 0 for x in y]
                print(self.W[i])
                for j in range(X.shape[0]):
                    if y_label[j] == 1:
                        for k in range(len(M)):
                            Fh.append(
                                self.T[X.shape[0] * i * len(M) + j * len(M) + k] ==
                                (LinearClassificationMultiClassLearner.sumproduct(list(X.iloc[j]), self.W[i]) > M[k]))
                            Fs.append(self.T[X.shape[0] * i * len(M) + j * len(M) + k])
                    else:
                        for k in range(len(M)):
                            Fh.append(
                                self.T[X.shape[0] * i * len(M) + j * len(M) + k] ==
                                (LinearClassificationMultiClassLearner.sumproduct(list(X.iloc[j]), self.W[i]) < -M[k]))
                            Fs.append(self.T[X.shape[0] * i * len(M) + j * len(M) + k])
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

    def learn_without_sade(self, X, y, M=None, relaxation=1, K=None, weight_limit=100, learner='maxl'):
        if M is None:
            print('please provide a list of margins')
            sys.exit(0)

        # for classification problems, bigger margin is stricter
        M.sort()
        self.targets = np.unique(y)

        # relaxation variable used in maxsat
        self.relaxation = relaxation

        # parameter initialization
        self.W = [[Real('w_{}_{}'.format(i, j)) for i in range((X.shape[1] + 1))] for j in range(len(np.unique(y)))]
        self.add_weight_constraints(low=-weight_limit, high=weight_limit)

        if K is not None:
            self.add_knowledge_constraints(X, y, K)
            print('knowledge constraints added')

        self.T = BoolVector('t', X.shape[0] * len(M) * len(np.unique(y)))
        self.add_decision_constraints(X, y, M=M, learner=learner)
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
            training_prediction = self.predict(X, y)
            print('training accuracy:', training_prediction[2])

    def learn(self, X, y, M=None, relaxation=1, K=None, weight_limit=100,
              validation_set=None, batch_size=20, epochs=20, learning_rate=0.1,
              learner='maxl', step_size=1, waiting_period=5, early_stopping=True):

        self.solver.set('timeout', waiting_period * 1000)
        early_stopping = early_stopping
        if M is None:
            print('please provide a list of margins')
            sys.exit(0)

        # for classification problems, bigger margin is stricter
        M.sort()
        self.targets = np.unique(y)

        # relaxation variable used in maxsat
        self.relaxation = relaxation

        # parameter initialization
        self.W = [[Real('w_{}_{}'.format(i, j)) for i in range((X.shape[1] + 1))] for j in range(len(np.unique(y)))]
        print(self.W)

        all_indices = list(range(X.shape[0]))
        all_batches = [all_indices[i * batch_size: (i + 1) * batch_size]
                       for i in range(int(len(all_indices) / batch_size))]
        current_weights = []
        gradient_direction = []
        stop_learning = False
        variable_learning_rate = learning_rate
        for ep in range(epochs):
            for j in range(len(all_batches)):
                if (len(self.training_losses) + 1) % 50 == 0:
                    print('\n\n iteration: {} \n\n'.format(len(self.training_losses) + 1))
                active_indices = all_batches[j]
                print('number of instances being considered:', len(active_indices))
                if K is not None:
                    if len(self.training_losses) % step_size == 0:
                        variable_learning_rate = learning_rate
                        self.add_knowledge_constraints(X, y, K)
                        print('knowledge constraints added, maximal step size is {}'.format(variable_learning_rate))
                    else:
                        variable_learning_rate = 0.05
                        print('knowledge constraints not added, maximal step size is {}'.format(variable_learning_rate))

                self.add_weight_constraints(low=-weight_limit, high=weight_limit)

                if len(current_weights) > 0:
                    print(gradient_direction)
                    self.Hard_Constraints.append(
                        And([And([If(current_weights[i][k] == -weight_limit, self.W[i][k] > current_weights[i][k],
                                     And(self.W[i][k] < current_weights[i][k], self.W[i][k] >= current_weights[i][k] -
                                         variable_learning_rate)) if gradient_direction[i][k] == 1
                                  else If(current_weights[i][k] == weight_limit, self.W[i][k] < current_weights[i][k],
                                          And(self.W[i][k] > current_weights[i][k], self.W[i][k] <=
                                              current_weights[i][k] + variable_learning_rate))
                                  for k in range(len(self.W[i]))]) for i in range(len(self.W))]))

                self.T = BoolVector('t', X.shape[0] * len(M) * len(np.unique(y)))
                X_sub = X.iloc[active_indices, :]
                y_sub = [y[i] for i in active_indices]
                X_sub.reset_index(drop=True, inplace=True)
                self.add_decision_constraints(X_sub, y_sub, M=M, learner=learner)

                # optimization step (maxsat)
                start = time.time()
                if learner == 'maxl':
                    self.OptimizationMaxL(length=len(M))
                elif learner == 'fumalik':
                    self.OptimizationFuMalik(length=len(M))
                else:
                    raise ValueError('Provide a valid learner, either maxl or fumalik')
                self.runtime = self.runtime + time.time() - start

                print('runtime:', self.runtime)

                if self.out == sat:
                    self.W_learnt = [
                        [self.solver.model()[u].numerator_as_long() / self.solver.model()[u].denominator_as_long()
                         for u in w] for w in self.W]
                    print('learned parameters')
                    print(self.W_learnt)
                    self.all_weights.append(self.W_learnt)
                    gradient_all = self.calculate_gradient(X, y)
                    gradient_values = gradient_all[0]
                    self.training_losses.append(gradient_all[1])
                    gradient_direction = [[1 if g >= 0 else -1 for g in G] for G in gradient_values]
                    print('gradient:', gradient_values)
                    current_weights = self.W_learnt.copy()
                    training_prediction = self.predict(X, y)
                    print('training accuracy:', training_prediction[2], self.training_losses[-1])
                    if validation_set is not None:
                        self.validation_losses.append(self.calculate_gradient(validation_set[0], validation_set[1])[1])
                        self.validation_accuracies.append(self.predict(validation_set[0], validation_set[1])[2])
                    self.training_accuracies.append(training_prediction[2])
                    if variable_learning_rate == 0.05:
                        self.knowledge_constraint_index.append(0)
                    else:
                        self.knowledge_constraint_index.append(1)
                else:
                    print('found unknown. randomizing gradients, moving to the next batch')
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
                            avg_loss_last_50 = statistics.mean(self.training_losses[-200:])
                            avg_loss_last_100_50 = statistics.mean(self.training_losses[-400:-200])
                            if (avg_loss_last_100_50 - avg_loss_last_50) / avg_loss_last_100_50 <= 0.02:
                                print('stopping the iterations as there is no improvement')
                                stop_learning = True
                                break
            if stop_learning:
                break

        self.W_learnt = current_weights
        print('all training accuracies:', self.training_accuracies)
        print('all validation accuracies:', self.validation_accuracies)
        if validation_set is not None:
            print('all validation loss:', self.validation_losses)
            self.W_learnt = self.all_weights[self.validation_accuracies.index(max(self.validation_accuracies))]
            # self.W_learnt = self.all_weights[self.validation_losses.index(min(self.validation_losses))]
        else:
            print('all training loss:', self.training_losses)
            self.W_learnt = self.all_weights[self.training_accuracies.index(max(self.training_accuracies))]
            # self.W_learnt = self.all_weights[self.training_losses.index(min(self.training_losses))]

    def get_plot(self, plot_suffix=''):
        plt.figure()
        plt.subplot(211)
        plt.plot(self.training_accuracies)
        plt.ylim(top=1)
        plt.ylim(bottom=0)
        plt.xlabel('batches')
        plt.ylabel('training accuracy')
        plt.subplot(212)
        plt.plot(self.training_losses)
        plt.legend(self.targets)
        plt.xlabel('batches')
        plt.ylabel('loss')

        if K is None:
            filename = 'training_accuracy_without_knowledge_' + plot_suffix + '.png'
        else:
            filename = 'training_accuracy_with_knowledge_' + plot_suffix + '.png'

        plt.savefig(filename)
        plt.close()

        if len(self.validation_losses) > 0:
            plt.figure()
            plt.subplot(211)
            plt.plot(self.validation_accuracies)
            plt.ylim(top=1)
            plt.ylim(bottom=0)
            plt.xlabel('batches')
            plt.ylabel('accuracy')
            plt.subplot(212)
            plt.plot(self.validation_losses)
            plt.legend(self.targets)
            plt.xlabel('batches')
            plt.ylabel('loss')

            if K is None:
                filename = 'validation_accuracy_without_knowledge_' + plot_suffix + '.png'
            else:
                filename = 'validation_accuracy_with_knowledge_' + plot_suffix + '.png'

            plt.savefig(filename)
            plt.close()

    def predict(self, X, y):
        weights = self.W_learnt
        indices = list(range(X.shape[0]))

        X_1 = X.copy()
        X_1.insert(loc=0, column='intercept', value=[1] * X.shape[0])
        X_1 = X_1[:].values

        confidence = [list(X_1.dot(pd.Series(weights[i]))) for i in range(len(self.targets))]
        predictions = [[confidence[j][i] for j in range(len(np.unique(y)))] for i in range(X.shape[0])]
        predictions = [np.unique(y)[p.index(max(p))] for p in predictions]
        predictions_correct = [1 if y[i] == predictions[i] else 0 for i in range(len(predictions))]

        return predictions, sum(predictions_correct), sum(predictions_correct) / len(predictions_correct), confidence

    def calculate_gradient(self, X, y):
        gradient = []
        loss = []
        self.instances_to_change = []
        if len(self.instances_to_change) > 0:
            X_val = X
            y_val = y.copy()
            for i in self.instances_to_change:
                if y[i] == 0:
                    y_val[i] = 1
                else:
                    y_val[i] = 0
        else:
            X_val = X
            y_val = y

        H = self.predict(X_val, y_val)[3]

        for i in range(len(self.targets)):
            g = []
            h = H[i]
            y_label = [1 if x == self.targets[i] else 0 for x in y]

            Z = [1 / (1 + np.exp(-h[j])) for j in range(len(y_val))]
            L = log_loss(y_label, Z)
            Z_Y = [Z[j] - y_label[j] for j in range(len(y_val))]

            g.append(sum(Z_Y))
            g.extend(list(X_val.T.dot(Z_Y)))
            g = [gr / X_val.shape[0] for gr in g]
            gradient.append(g)
            loss.append(L)

        return gradient, loss
