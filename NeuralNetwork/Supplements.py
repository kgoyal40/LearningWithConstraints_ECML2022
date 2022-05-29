from sklearn.preprocessing import MinMaxScaler
from LinearClassifierLoanEligibility import *
import pandas as pd
import statistics


def data_processing(data):
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    categorical_features = list(data.select_dtypes(include='object').columns)
    categorical_features = list(set(categorical_features))
    numerical_features = [c for c in data.columns if c not in categorical_features]
    print(categorical_features)
    print(numerical_features)
    scaler = MinMaxScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    for _c in categorical_features:
        data[_c] = pd.Categorical(data[_c])
    df_transformed = pd.get_dummies(data, drop_first=True)
    return df_transformed, scaler


def data_manipulation(X, y):
    y_violated = y.copy()
    indices_to_manipulate = [i for i in range(len(y)) if X['Credit_History'][i] == 0]
    indices_to_manipulate = random.sample(indices_to_manipulate, 35)
    for i in indices_to_manipulate:
        y_violated[i] = 1
    return X, y_violated


def find_counter_example(X, parameters=None, K=None, num=10):
    solutions = []
    Fs = []
    if parameters is None or len(parameters) != X.shape[1] + 1:
        print('please provide a valid set of parameters')
        sys.exit()

    s = Solver()
    _X = RealVector('x', X.shape[1])

    for i in range(0, 4):
        Fs.append(And(_X[i] <= 1, _X[i] > 0))

    for i in range(4, 14):
        Fs.append(Or(_X[i] == 0, _X[i] == 1))

    Fs.append(Not(And(_X[12] == 1, _X[13] == 1)))
    Fs.append(PbLe([(_X[7] == 1, 1), (_X[8] == 1, 1), (_X[9] == 1, 1)], 1))

    if K is not None:
        print('adding knowledge constraints')
        Fs.append(And(_X[4] == 0, _X[0] <= K[0], LinearClassificationLearner.sumproduct(_X, parameters) > 0))

    for j in range(num):
        out = s.check(Fs)
        if out == sat:
            solution = [s.model()[x].numerator_as_long() / s.model()[x].denominator_as_long() for x in _X]
            solutions.append([solution, LinearClassificationLearner.sumproduct(solution, parameters)])
            Fs.append(Or([_X[i] != s.model()[_X[i]] for i in range(len(_X))]))
            s.reset()
        else:
            print(j, 'solutions found')
            break

    return solutions


def count_violations(X, y, income):
    indices = []
    count = 0
    for i in range(X.shape[0]):
        if X['Credit_History'][i] == 0 and X['ApplicantIncome'][i] <= income:
            indices.append(i)
            if y[i] == 1:
                count = count + 1
    return count, indices


def calculate_modified_accuracy(X, y, y_pred, K):
    count = 0
    correct_predictions = 0
    for i in range(X.shape[0]):
        if not(X['Credit_History'][i] == 0 and X['ApplicantIncome'][i] <= K[0] and y[i] == 1):
            count += 1
            if y[i] == y_pred[i]:
                correct_predictions += 1
    return correct_predictions/count


def get_best_alpha(baseline_results):
    alphas = np.unique(list(baseline_results['margin/alpha']))
    processed_baseline_results = pd.DataFrame()

    for a in alphas:
        baseline_results_sub = baseline_results[baseline_results['margin/alpha'] == a]
        mean_accuracy_test = statistics.mean(list(baseline_results_sub['accuracy_test_corrected']))
        std_accuracy_test = statistics.stdev(list(baseline_results_sub['accuracy_test_corrected']))

        mean_test_violations = statistics.mean(list(baseline_results_sub['violations_test_prediction']))
        std_test_violations = statistics.stdev(list(baseline_results_sub['violations_test_prediction']))

        processed_baseline_results = processed_baseline_results.append([[a, mean_accuracy_test, std_accuracy_test,
                                                                         mean_test_violations, std_test_violations]])

    best_alpha = min(processed_baseline_results.values.tolist(), key=lambda x: x[-2])[0]
    return best_alpha
