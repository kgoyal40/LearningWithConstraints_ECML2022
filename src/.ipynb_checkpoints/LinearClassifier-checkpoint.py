from Optimizer import Optimizer
from z3 import *
import time
import random
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, accuracy_score
import pandas as pd
import numpy as np
import statistics


class LinearClassificationLearner(Optimizer):
    """
    Basic Linear Classification happens in this class. Inherits the optimizer class where the magic happens
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
        self.K = False
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
            for i in range(X.shape[0]):
                if y[i] == 1:
                    for j in range(len(M)):
                        Fh.append(
                            self.T[i * len(M) + j] == (LinearClassificationLearner.sumproduct(list(X.iloc[i]), self.W) >
                                                       M[j]))
                    Fs.append(Sum([If(self.T[i * len(M) + j], 1, 0) for j in range(len(M))]) >= len(M))
                else:
                    for j in range(len(M)):
                        Fh.append(
                            self.T[i * len(M) + j] == (LinearClassificationLearner.sumproduct(list(X.iloc[i]), self.W)
                                                       < -1 * M[j]))
                    Fs.append(Sum([If(self.T[i * len(M) + j], 1, 0) for j in range(len(M))]) >= int(len(M)))
        elif learner == 'fumalik':
            for i in range(X.shape[0]):
                if y[i] == 1:
                    for j in range(len(M)):
                        Fh.append(
                            self.T[i * len(M) + j] == (LinearClassificationLearner.sumproduct(list(X.iloc[i]), self.W) >
                                                       M[j]))
                        Fs.append(self.T[i * len(M) + j])
                else:
                    for j in range(len(M)):
                        Fh.append(
                            self.T[i * len(M) + j] == (LinearClassificationLearner.sumproduct(list(X.iloc[i]), self.W)
                                                       < -1 * M[j]))
                        Fs.append(self.T[i * len(M) + j])
        else:
            raise ValueError('Provide a valid learner, either maxl or fumalik')

        self.Soft_Constraints.extend(Fs)
        self.Hard_Constraints.extend(Fh)

    def add_weight_constraints(self, low=-1000, high=1000):
        Fh = []
        for w in self.W:
            Fh.append(And(w >= low, w <= high))
        self.Hard_Constraints.extend(Fh)

    def learn_without_sade(self, X, y, M=None, relaxation=1, K=None, weight_limit=100, learner='maxl'):
        if K is not None:
            self.K = True

        if M is None:
            print('please provide a list of margins')
            sys.exit(0)

        # for classification problems, bigger margin is stricter
        M.sort()

        # relaxation variable used in maxsat
        self.relaxation = relaxation

        # parameter initialization
        self.W = RealVector('w', X.shape[1] + 1)

        if K is not None:
            self.add_knowledge_constraints(X, y, K)
            print('knowledge constraints added')

        self.add_weight_constraints(low=-weight_limit, high=weight_limit)

        self.T = BoolVector('t', X.shape[0] * len(M))

        self.add_decision_constraints(X, y, M=M, learner=learner)

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
            self.W_learnt = [self.solver.model()[w].numerator_as_long() /
                             self.solver.model()[w].denominator_as_long() for w in self.W]
            print('learned parameters')
            print(self.W_learnt)
            training_prediction = self.predict(X, y)
            print('training accuracy:', training_prediction[2])

    def learn(self, X, y, M=None, relaxation=1, K=None, weight_limit=100,
              validation_set=None, batch_size=50, epochs=1, learning_rate=0.1,
              learner='maxl', step_size=5, waiting_period=10, early_stopping=True):

        # step_size parameter controls the iterations where the knowledge constraint is enforced

        self.solver.set('timeout',  waiting_period*1000)
        early_stopping = early_stopping
        if K is not None:
            self.K = True

        if M is None:
            print('please provide a list of margins')
            sys.exit(0)

        # for classification problems, bigger margin is stricter
        M.sort()

        # relaxation variable used in maxsat
        self.relaxation = relaxation

        # parameter initialization
        self.W = RealVector('w', X.shape[1] + 1)

        all_indices = list(range(X.shape[0]))
        all_batches = [all_indices[i * batch_size: (i + 1) * batch_size]
                       for i in range(int(len(all_indices) / batch_size))]
        current_weights = []
        gradient_direction = []
        variable_learning_rate = learning_rate
        self.knowledge_constraint_index = []

        # random initial weights don't work, i tried and failed
        stop_learning = False
        for ep in range(epochs):
            for j in range(len(all_batches)):
                active_indices = all_batches[j]
                print('number of instances being considered:', len(active_indices))
                X_sub = X.iloc[active_indices, :]
                y_sub = [y[i] for i in active_indices]
                X_sub.reset_index(drop=True, inplace=True)
                if K is not None:
                    if len(self.training_losses) % step_size == 0:
                        self.add_knowledge_constraints(X, y, K)
                        variable_learning_rate = learning_rate
                        print('knowledge constraints added, learning rate {}'.format(variable_learning_rate))
                    else:
                        variable_learning_rate = 0.05
                        print('knowledge constraints not added, learning rate {}'.format(variable_learning_rate))

                self.add_weight_constraints(low=-weight_limit, high=weight_limit)

                if len(current_weights) > 0:
                    print(gradient_direction)
                    self.Hard_Constraints.append(
                        And([If(current_weights[i] == -weight_limit, True,
                                And(self.W[i] < current_weights[i], self.W[i] >= current_weights[i] -
                                    variable_learning_rate)) if gradient_direction[i] == 1
                             else If(current_weights[i] == weight_limit, True,
                                     And(self.W[i] > current_weights[i], self.W[i] <= current_weights[i] +
                                         variable_learning_rate)) for i in range(len(self.W))]))

                self.T = BoolVector('t', len(active_indices) * len(M))
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

                if self.out == sat:
                    self.W_learnt = [self.solver.model()[w].numerator_as_long() /
                                     self.solver.model()[w].denominator_as_long() for w in self.W]
                    print('learned parameters')
                    print(self.W_learnt)
                    self.all_weights.append(self.W_learnt)
                    gradient_all = self.calculate_gradient(X, y, self.W_learnt)
                    gradient_values = gradient_all[0]
                    self.training_losses.append(gradient_all[1])
                    gradient_direction = [1 if g >= 0 else -1 for g in gradient_values]
                    print('gradient:', gradient_values)
                    current_weights = self.W_learnt.copy()
                    print('runtime:', self.runtime)
                    training_prediction = self.predict(X, y)
                    print('training accuracy:', training_prediction[2])
                    if validation_set is not None:
                        self.validation_losses.append(self.calculate_gradient(validation_set[0], validation_set[1],
                                                                              self.W_learnt)[1])
                        self.validation_accuracies.append(self.predict(validation_set[0], validation_set[1])[2])

                    self.training_accuracies.append(training_prediction[2])
                    if variable_learning_rate == 0.05:
                        self.knowledge_constraint_index .append(0)
                    else:
                        self.knowledge_constraint_index .append(1)

                else:
                    print('found unknown. randomizing gradients')
                    gradient_direction = [g * -1 for g in gradient_direction]

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
        print('all training loss:', self.training_losses)

        if validation_set is not None:
            self.W_learnt = self.all_weights[self.validation_losses.index(min(self.validation_losses))]
        else:
            self.W_learnt = self.all_weights[self.training_losses.index(min(self.training_losses))]

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
        plt.xlabel('batches')
        plt.ylabel('training entropy loss')

        if self.K is None:
            filename = 'training_set_without_knowledge_' + plot_suffix + '.png'
        else:
            filename = 'training_set_with_knowledge_' + plot_suffix + '.png'

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
            plt.xlabel('batches')
            plt.ylabel('entropy loss')

            if self.K is None:
                filename = 'validation_set_without_knowledge_' + plot_suffix + '.png'
            else:
                filename = 'validation_set_with_knowledge_' + plot_suffix + '.png'

            plt.savefig(filename)
            plt.close()

    def calculate_gradient(self, X, y, weights):
        gradient = []
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

        h = self.predict(X_val, y_val, weights=weights)[3]
        Z = [1 / (1 + np.exp(-h[i])) for i in range(len(y_val))]
        loss = log_loss(y_val, Z)
        Z_Y = [Z[i] - y_val[i] for i in range(len(y_val))]
        gradient.append(sum(Z_Y))
        gradient.extend(list(X_val.T.dot(Z_Y)))
        gradient = [g / X_val.shape[0] for g in gradient]

        return gradient, loss

    def predict(self, X, y, weights=None):
        if weights is None:
            weights = self.W_learnt

        X_1 = X.copy()
        X_1.insert(loc=0, column='intercept', value=[1] * X.shape[0])
        X_1 = X_1[:].values
        confidence = X_1.dot(pd.Series(weights))
        predictions = [1 if c > 0 else 0 for c in confidence]

        predictions_correct = [1 if y[i] == predictions[i] else 0 for i in range(len(predictions))]

        if len(self.instances_to_change) > 0:
            predictions_correct = [1 if i in self.instances_to_change else predictions_correct[i]
                                   for i in range(len(predictions_correct))]

        return predictions, sum(predictions_correct), sum(predictions_correct) / len(predictions_correct), confidence
