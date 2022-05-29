from z3 import *


class Optimizer:
    """
    This is a general purpose Optimization with a MaxSAT approach that takes into account multiple margins and tries
    to satisfy the strict constraints on a decision point first.

    W: parameters to learn
    T: Boolean variables for training instances. One training instance has M T variables. M is the number of margins.
       For each instance, first constraint is the most lenient and the last constraint is the most strict.
    Soft Constraints: Constraints based on training instances. For each instance there is one constraint.
    Hard Constraints: By default constraint the definitions of each constraint in T. Specific knowledge constraints
                      are also added for specific use cases

    Optimization function is the main MaxSAT algorithm.
    """

    def __init__(self, verbose=False):
        # t = Then('simplify', 'propagate-values', 'qe-light', 'simplify', 'nlqsat', 'smt')
        # self.solver = t.solver()
        # self.solver.set(unsat_core=True)

        self.solver = SolverFor('NRA')
        # self.solver = Solver()
        # set_option('macro-finder', True)
        # set_option('ignore_patterns_on_ground_qbody', False)
        self.W = None
        self.T = None
        self.Soft_Constraints = []
        self.Hard_Constraints = []
        self.relaxation = None
        self.out = None
        self.solver.set('timeout',  10000)

        if verbose:
            set_option('verbose', 10)

        set_param('smt.arith.solver', 2)
        # set_param('smt.relevancy', 1)

    def reset(self):
        self.solver.reset()
        self.W = None
        self.T = None
        self.Soft_Constraints = []
        self.Hard_Constraints = []
        self.relaxation = None
        self.out = None

    @staticmethod
    def GetLhsRhs(c):
        split = c.split(' >=')
        return split[0], split[1][-1]

    @staticmethod
    def GetClauseNum(c):
        return int(c.split(',')[0].split('__')[1])

    def OptimizationMaxL(self, length=None):
        print('starting the check')
        self.solver.add(self.Hard_Constraints)
        if length is None:
            print('please provide the total number of margins')
            sys.exit(0)
        i = 0
        Fs = self.Soft_Constraints.copy()

        while True:
            out = self.solver.check(Fs)
            if out == sat:
                print('found sat')
                self.out = sat
                break
            elif out == unknown:
                print('found unknown')
                self.out = unknown
                break
            else:
                print('found unsat')
                relaxed_variables = []
                core = self.solver.unsat_core()
                print('size of core:', len(core))
                for c in core:
                    Fs.remove(c)
                    if c.__str__()[0] == 'O':
                        i += 1
                        relaxed_variables.append(Bool('r_' + str(i)))
                        Fs.append(Or(c, Bool('r_' + str(i))))
                    else:
                        split = Optimizer.GetLhsRhs(c.__str__())
                        clause_id = Optimizer.GetClauseNum(c.__str__())
                        if int(split[1]) > 1:
                            Fs.append(Sum([If(Bool('t__' + str(clause_id + j)), 1, 0) for j in range(length)]) >=
                                      int(split[1]) - 1)
                        else:
                            i += 1
                            relaxed_variables.append(Bool('r_' + str(i)))
                            Fs.append(Or(Bool('t__' + str(clause_id)), Bool('r_' + str(i))))

                if len(relaxed_variables) > 0:
                    self.solver.add(Sum([If(r, 1, 0) for r in relaxed_variables]) == self.relaxation)
                if len(core) == 0:
                    print('no solution is possible')
                    self.out = unsat
                    break

    def OptimizationFuMalik(self, length=None):
        print('starting the check')
        self.solver.add(self.Hard_Constraints)
        if length is None:
            print('please provide the total number of margins')
            sys.exit(0)
        i = 0
        Fs = self.Soft_Constraints.copy()
        while True:
            out = self.solver.check(Fs)
            if out == sat:
                print('found sat')
                self.out = sat
                break
            elif out == unknown:
                print('found unknown')
                self.out = unknown
                break
            else:
                print('found unsat')
                relaxed_variables = []
                core = self.solver.unsat_core()
                print('size of core:', len(core))
                for c in core:
                    i += 1
                    Fs.remove(c)
                    Fs.append(Or(c, Bool('r_' + str(i))))
                if len(relaxed_variables) > 0:
                    self.solver.add(Sum([If(r, 1, 0) for r in relaxed_variables]) == self.relaxation)
                if len(core) == 0:
                    print('no solution is possible')
                    self.out = unsat
                    break
