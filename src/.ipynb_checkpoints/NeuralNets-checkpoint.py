import numpy as np
from z3 import *
from Optimizer import Optimizer
from sklearn.model_selection import train_test_split
import pandas as pd
from LinearClassifier import *
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, accuracy_score


class NeuralNetTorch(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes
        self.Layers = []
        for i in range(len(self.sizes) - 1):
            self.Layers.append(nn.Linear(self.sizes[1 + i - 1], self.sizes[1 + i]))

    def forward(self, X):
        out = X
        for i in range(len(self.Layers)):
            out = self.Layers[i](out)
            if i != len(self.Layers) - 1:
                out = F.relu(out)
        return out


def relu(vector):
    output = []
    for i in range(len(vector)):
        temp = []
        for j in range(len(vector[i])):
            temp.append(If(vector[i][j] > 0, vector[i][j], 0))
        output.append(np.array(temp))
    return np.array(output)


def sigmoid(vector):
    output = []
    for i in range(len(vector)):
        temp = []
        for j in range(len(vector[i])):
            temp.append(1/(1 + pow(2.71, vector[i][j])))
        output.append(np.array(temp))
    return np.array(output)


class LinearLayer:
    # only relu and sigmoid  activation is supported right now
    def __init__(self, layer_name=None, in_size=10, out_size=1, activation='None'):
        if layer_name is None:
            raise Exception('Please Provide a Name for the Layer')
        else:
            self.layer_name = layer_name
            self.in_size = in_size
            self.out_size = out_size
            self.W = np.array([np.array(RealVector(layer_name + '__{}'.format(i), in_size + 1))
                               for i in range(out_size)])
            self.W_learnt = None
            self.output = None
            self.activation = activation
            print('Layer: {}; size: [{} {}]; Activation: {}'.format(self.layer_name, self.in_size,
                                                                    self.out_size, self.activation))

    def compute(self, X):
        self.output = []
        print(X)
        sys.exit(0)
        for i in range(len(X)):
            if len(X[i]) != self.in_size:
                raise Exception("data input size doesn't match the layer input size")
            else:
                temp = []
                for j in range(self.out_size):
                    temp.append(Sum(list(np.insert(X[i], 0, 1, axis=0) * self.W[j])))
                self.output.append(np.array(temp))
        self.output = np.array(self.output)
        if self.activation == 'Relu':
            self.output = relu(np.array(self.output))
        elif self.activation == 'Sigmoid':
            self.output = sigmoid(np.array(self.output))


class NeuralNet(Optimizer):
    def __init__(self, sizes=None):
        super().__init__()
        if sizes is None or len(sizes) < 2:
            raise Exception('Please provide correct list of input sizes')
        else:
            self.sizes = sizes
            self.Layers = []
            for i in range(len(sizes) - 1):
                if i != len(sizes) - 2:
                    self.Layers.append(LinearLayer('l{}'.format(i + 1), sizes[i], sizes[i + 1],
                                                   activation='Relu'))
                else:
                    self.Layers.append(LinearLayer('l{}'.format(i + 1), sizes[i], sizes[i + 1],
                                                   activation='None'))
            self.W = []
            self.training_accuracies = []
            self.training_losses = []

    def forward(self, X):
        output_data = X
        for i in range(len(self.Layers)):
            self.Layers[i].compute(output_data)
            output_data = self.Layers[i].output
        return output_data

    def add_knowledge_constraints(self, X, y, K=None):
        pass

    def add_decision_constraints(self, X, y):
        Fs = []
        Fh = []
        Predictions = self.forward(X)
        for i in range(X.shape[0]):
            if y[i] == 1:
                Fh.append(self.T[i] == (Predictions[i][0] > 0))
            else:
                Fh.append(self.T[i] == (Predictions[i][0] < 0))
            Fs.append(self.T[i])

        self.Soft_Constraints.extend(Fs)
        self.Hard_Constraints.extend(Fh)

    def add_weight_constraints(self, low=-100, high=100):
        print('adding weight constraints')
        self.Hard_Constraints.append(And([And([And([And(self.Layers[i].W[j][k] >= low, self.Layers[i].W[j][k] <= high)
                                                    for k in range(self.Layers[i].in_size + 1)])
                                               for j in range(self.Layers[i].out_size)])
                                          for i in range(len(self.Layers))]))

    def learn(self, X, y, K=None, batch_size=5, epochs=5, learning_rate=0.001):
        self.relaxation = 1

        all_indices = list(range(X.shape[0]))
        random.shuffle(all_indices)
        all_batches = [all_indices[i * batch_size: (i + 1) * batch_size]
                       for i in range(int(len(all_indices) / batch_size))]

        gradients = []
        for e in range(epochs):
            for b in range(len(all_batches)):
                active_indices = all_batches[b]
                print('number of instances being considered:', len(active_indices))
                X_sub = X[active_indices]
                y_sub = [y[i] for i in active_indices]

                if K is not None:
                    self.add_knowledge_constraints(X, y, K)
                    print('knowledge constraints added')

                if len(self.W) > 0:
                    self.Hard_Constraints.append(And([And([And([And(self.Layers[i].W[j][k] <= self.W[i][j][k],
                                                                    self.Layers[i].W[j][k] >= self.W[i][j][k] -
                                                                    learning_rate)
                                                                if gradients[i][j][k] > 0 else
                                                                And(self.Layers[i].W[j][k] <= self.W[i][j][k] +
                                                                    learning_rate,
                                                                    self.Layers[i].W[j][k] >= self.W[i][j][k])
                                                                for k in range(self.Layers[i].in_size + 1)])
                                                           for j in range(self.Layers[i].out_size)])
                                                      for i in range(len(self.Layers))]))

                self.T = BoolVector('t', len(active_indices))
                self.add_weight_constraints(low=-100, high=100)
                self.add_decision_constraints(X_sub, y_sub)
                self.OptimizationFuMalik(length=1)

                if self.out == sat:
                    self.get_weights()
                    print(self.W)
                    predictions, gradients, loss, accuracy = self.predict(X, y)
                    print('training accuracy:', accuracy)
                    self.training_accuracies.append(accuracy)
                    self.training_losses.append(loss)
                else:
                    print('found unknown. reversing gradients')
                    gradients = [[[gradients[i][j][k] * -1 for k in range(self.Layers[i].in_size + 1)]
                                  for j in range(self.Layers[i].out_size)]
                                 for i in range(len(self.Layers))]
                self.Soft_Constraints = []
                self.Hard_Constraints = []
                self.T = None
                self.solver.reset()
                self.out = None

    def get_weights(self):
        self.W = [[[self.solver.model()[L.W[i][j]].numerator_as_long() /
                    self.solver.model()[L.W[i][j]].denominator_as_long() if self.solver.model()[L.W[i][j]] is not None
                    else 0 for j in range(L.in_size + 1)] for i in range(L.out_size)]
                  for L in self.Layers]

    def predict(self, X, y):
        mirror_model = NeuralNetTorch(self.sizes)
        for i in range(len(mirror_model.Layers)):
            mirror_model.Layers[i].weight = nn.Parameter(torch.Tensor([w[1:] for w in self.W[i]]))
            mirror_model.Layers[i].bias = nn.Parameter(torch.Tensor([w[0] for w in self.W[i]]))
        out = mirror_model(torch.Tensor(X))
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(out, torch.Tensor([[v] for v in y]).type(torch.FloatTensor))
        loss.backward()
        gradients_bias = [mirror_model.Layers[i].bias.grad.tolist() for i in range(len(mirror_model.Layers))]
        gradients_weights = [mirror_model.Layers[i].weight.grad.tolist() for i in range(len(mirror_model.Layers))]
        gradients = [[[gradients_bias[i][j]] + gradients_weights[i][j]
                      for j in range(len(gradients_weights[i]))]
                     for i in range(len(mirror_model.Layers))]
        y_pred = [1 if o[0] > 0 else 0 for o in out.tolist()]
        accuracy = accuracy_score(y, y_pred)
        return out, gradients, loss.detach().cpu().numpy().item(), accuracy
