from sklearn.preprocessing import MinMaxScaler
from LinearClassifierMusic import *
import pandas as pd
from CrossValidationMultiClass import *
import statistics


def count_violations(X, y):
    count = 0
    for i in range(X.shape[0]):
        if X['group_The Beatles'][i] == 1:
            if y[i] in ['Electronic', 'Metal', 'Classical']:
                count += 1
    return count


def find_counter_example(X, parameters=None, K=1, num=10):
    solutions = []
    Fs = []

    s = Solver()
    _X = RealVector('x', X.shape[1])

    for i in [2, 3]:
        Fs.append(Or([_X[i] == v for v in np.unique(list(X.iloc[:, i]))]))

    for i in [0, 1]:
        Fs.append(And(_X[i] <= max(list(X.iloc[:, i])), _X[i] >= min(list(X.iloc[:, i]))))

    for i in range(4, 9):
        Fs.append(_X[i] == 0)

    if K is not None:
        print('adding knowledge constraints')
        Fs.append(And(_X[9] == 1, Or([MusicLinearClassifier.sumproduct(_X, parameters[i]) > 0 for i in range(3)])))

    for j in range(num):
        out = s.check(Fs)
        if out == sat:
            solution = [s.model()[x].numerator_as_long() / s.model()[x].denominator_as_long() for x in _X]
            solutions.append(solution)
            Fs.append(Or([_X[i] != s.model()[_X[i]] for i in range(len(_X))]))
            s.reset()
        else:
            print(j, 'solutions found')
            break

    return solutions


def find_counter_example_2(X, parameters=None, K=1, num=10):
    solutions = []
    Fs = []

    s = Solver()
    _X = RealVector('x', X.shape[1])

    for i in [0, 1, 2, 3, 4, 5]:
        Fs.append(And(_X[i] <= max(list(X.iloc[:, i])), _X[i] >= min(list(X.iloc[:, i]))))

    for i in range(6, 12):
        Fs.append(_X[i] == 0)

    if K is not None:
        print('adding knowledge constraints')
        Fs.append(And(_X[12] == 1, Or([MusicLinearClassifier.sumproduct(_X, parameters[i]) > 0 for i in range(3)])))

    for j in range(num):
        out = s.check(Fs)
        if out == sat:
            solution = [s.model()[x].numerator_as_long() / s.model()[x].denominator_as_long() for x in _X]
            solutions.append(solution)
            Fs.append(Or([_X[i] != s.model()[_X[i]] for i in range(len(_X))]))
            s.reset()
        else:
            print(j, 'solutions found')
            break

    return solutions


def calculate_modified_accuracy(X, y, y_pred):
    count = 0
    correct_predictions = 0
    for i in range(X.shape[0]):
        if (X['group_The Beatles'][i] == 0) or ((X['group_The Beatles'][i] == 1) and (y[i] in ['Pop', 'Rock'])):
            count += 1
            if y[i] == y_pred[i]:
                correct_predictions += 1
    return correct_predictions/count


def get_best_alpha(baseline_results):
    alphas = [0, 1, 2, 5, 10, 50, 100]
    # alphas = [1, 50, 100]

    processed_baseline_results = pd.DataFrame()

    for a in alphas:
        baseline_results_sub = baseline_results[baseline_results['margin/alpha'] == a]
        mean_accuracy_test = statistics.mean(list(baseline_results_sub['accuracy_test_corrected']))
        std_accuracy_test = statistics.stdev(list(baseline_results_sub['accuracy_test_corrected']))

        mean_test_violations = statistics.mean(list(baseline_results_sub['violations_test_prediction']))
        std_test_violations = statistics.stdev(list(baseline_results_sub['violations_test_prediction']))

        processed_baseline_results = processed_baseline_results.append([[a, mean_accuracy_test, std_accuracy_test,
                                                                         mean_test_violations, std_test_violations]])

    processed_baseline_results.columns = ['alpha', 'mean accuracy', 'stddev accuracy',
                                          'mean violations', 'stddev violations']

    best_alpha = min(processed_baseline_results.values.tolist(), key=lambda x: x[-2])[0]
    return best_alpha
