import sys

sys.path.insert(0, '/Users/kshitijgoyal/Desktop/LearningWithConstraints_IJCAI/src')
from NeuralNets import *
import time
import torch


class NeuralNetLoanEligibility(NeuralNet):
    def __init__(self, sizes=None):
        super().__init__(sizes)
        self.income = None
        self.alpha = None

    def add_knowledge_constraints(self, X, y, K=None):
        if K is not None:
            self.income = K[0]
            self.alpha = K[1]
            print('adding knowledge constraint')
            _X = RealVector('_x', X.shape[1])
            # print(self.forward([_X]))
            print(ForAll(_X, Implies(And([_X[4] == 0] +
                                         [_X[0] <= self.income] +
                                         [And(_X[i] <= 1,
                                              _X[i] >= 0)
                                          for i in range(X.shape[1])]),
                                     self.forward([_X])[0][0] < 0)))
            self.Hard_Constraints.append(ForAll(_X, Implies(And([_X[4] == 0] +
                                                                [_X[0] <= self.income] +
                                                                [And(_X[i] <= 1,
                                                                     _X[i] >= 0)
                                                                 for i in range(X.shape[1])]),
                                                            self.forward([_X])[0][0] < 0)))
            # self.Hard_Constraints.append(Sum([If(w > 0, w, -1*w) for w in self.Layers[1].W[0]]) < 1)

            # t = Then('simplify', 'nra')
            # s = t.solver()
            # # s = Solver()
            # s.add(ForAll(_X, Implies(And([_X[4] == 0] + [_X[0] <= self.income] +
            #                              [And(_X[i] <= 1,
            #                                   _X[i] >= 0)
            #                               for i in range(X.shape[1])]),
            #                          self.forward([_X])[0][0] < 0)))
            # # s.add(And([And(w <= 1000, w >= -1000) for w in self.Layers[1].W[0]]))
            # print(s.check())
            # if s.check() == sat:
            #     print(s.model())
            #
            # sys.exit(0)
