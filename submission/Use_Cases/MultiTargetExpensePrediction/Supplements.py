from sklearn.preprocessing import MinMaxScaler
from BoundedMultiTargetRegression import *
import pandas as pd
import statistics
from sklearn.metrics import mean_squared_error


def find_counter_example(X, parameters=None, K=None, num=10):
    solutions = []
    Fs = []
    if parameters is None:
        print('please provide a valid set of parameters')
        sys.exit()

    s = Solver()
    _X = RealVector('x', X.shape[1])

    for i in [0, 1, 2, 5, 6, 7, 8, 9, 11]:
        Fs.append(Or([_X[i] == v for v in np.unique(list(X.iloc[:, i]))]))

    for i in [3, 4, 10, 12]:
        Fs.append(And(_X[i] <= 1, _X[i] >= 0))

    if K is not None:
        print('adding knowledge constraints')
        Fs.append(Or(Sum([MultiLinearRegressionLearner.sumproduct(_X, parameters[j])
                          for j in range(len(parameters))]) > ((_X[-1] - K[0]) / K[1]) + 0.01,
                     MultiLinearRegressionLearner.sumproduct(_X, parameters[1]) >
                     (0.05 * (_X[-1] - K[0]) / K[1]) + 0.01))

    for j in range(num):
        out = s.check(Fs)
        if out == sat:
            solution = [s.model()[x].numerator_as_long() / s.model()[x].denominator_as_long() for x in _X]
            solutions.append([solution, sum([MultiLinearRegressionLearner.sumproduct(solution, parameters[j])
                                             for j in range(len(parameters))])])
            Fs.append(Or([_X[i] != s.model()[_X[i]] for i in range(len(_X))]))
            s.reset()
        else:
            print(j, 'solutions found')
            break

    return solutions


def calculate_constraint_penalty(X, y_pred, K):
    transformed_household_income = (X['Total Household Income'] - K[0]) / K[1]
    Penalty_Constraint_1 = 0
    Penalty_Constraint_2 = 0
    for i in range(X.shape[0]):
        pred = list(map(lambda x: x[i], y_pred))
        Penalty_Constraint_1 += max(0, sum(pred) - transformed_household_income[i])
        Penalty_Constraint_2 += max(0, pred[1] - 0.05 * transformed_household_income[i])
    return Penalty_Constraint_1/X.shape[0] + Penalty_Constraint_2/X.shape[0]


def calculate_custom_metric_v0(X, y, y_pred, K):
    transformed_household_income = (X['Total Household Income'] - K[0]) / K[1]
    mse = 0
    for i in range(X.shape[0]):
        true = list(map(lambda x: x[i], y))
        pred = list(map(lambda x: x[i], y_pred))
        if (sum(true) > transformed_household_income[i]) or (true[1] > 0.05 * transformed_household_income[i]):
            Penalty_Constraint_1 = max(0, sum(pred) - transformed_household_income[i])
            Penalty_Constraint_2 = max(0, pred[1] - 0.05 * transformed_household_income[i])
            mse += (Penalty_Constraint_1**2 + Penalty_Constraint_2**2)
        else:
            mse += mean_squared_error(true, pred)*len(pred)
    return mse/X.shape[0]


def calculate_custom_metric_v1(X, y, y_pred, K):
    transformed_household_income = (X['Total Household Income'] - K[0]) / K[1]
    mse = 0
    for i in range(X.shape[0]):
        true = list(map(lambda x: x[i], y))
        pred = list(map(lambda x: x[i], y_pred))
        if sum(true) > transformed_household_income[i]:
            mse += (sum(pred) - transformed_household_income[i])**2
            if true[1] > 0.05 * transformed_household_income[i]:
                mse += (pred[1] - 0.05 * transformed_household_income[i]) ** 2
        else:
            mse += mean_squared_error(true, pred) * len(pred)
    return mse / X.shape[0]


def calculate_custom_metric(X, y, y_pred, K):
    transformed_household_income = (X['Total Household Income'] - K[0]) / K[1]
    mse = 0
    count = 0
    for i in range(X.shape[0]):
        true = list(map(lambda x: x[i], y))
        pred = list(map(lambda x: x[i], y_pred))
        if (sum(true) <= transformed_household_income[i]) and (true[1] <= 0.05 * transformed_household_income[i]):
            mse += mean_squared_error(true, pred) * len(pred)
            count += 1
    return mse / count


def count_violations(X, y, K=None):
    count = 0
    violated_instances = pd.DataFrame()
    for i in range(X.shape[0]):
        if (sum([y[j][i] for j in range(len(y))]) > ((X['Total Household Income'][i] - K[0]) / K[1]) + 0.01) or \
                (y[1][i] > ((0.05 * (X['Total Household Income'][i] - K[0])) / K[1]) + 0.01):
            count = count + 1
            violated_instances = violated_instances.append([list(X.iloc[i, :])])
    return count, violated_instances
