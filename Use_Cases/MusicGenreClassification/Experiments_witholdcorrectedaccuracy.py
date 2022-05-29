from Supplements import *
import sys
from scoop import futures
from MusicGenreClassificationRegularization import *
from sklearn.metrics import accuracy_score
import pickle
from sklearn.model_selection import train_test_split


def single_model_run(X, y, test_data=None, params=None, fold_index=None):
    if fold_index is None:
        print('provide the fold index!')
        sys.exit(0)

    output = pd.DataFrame()
    if params is None:
        print('provide valid set of parameters')
        sys.exit(0)
    else:
        if all([len(params[p]) == 1 for p in params]):
            print('no cross validation because single set of parameters')
            M, K, learning_rate, epoch, batch_size = params['M'][0], params['K'][0], params['learning_rate'][0], \
                                                     params['epoch'][0], params['batch_size'][0]
            CV_log = pd.DataFrame()
            val_evaluation = pd.DataFrame()
        else:
            all_cv_models, val_evaluation, best_parameter, CV_log = \
                cross_validate_multiclass(X, y, params=params)
            M, K, learning_rate, epoch, batch_size, validation_loss = best_parameter
            val_evaluation['outer_fold'] = fold_index

        while True:
            final_model = MusicLinearClassifier()
            final_model.learn(X, y, M=M, K=K, validation_set=None,
                              batch_size=batch_size, epochs=epoch, learner='fumalik',
                              learning_rate=learning_rate, step_size=1,
                              waiting_period=5, early_stopping=True)
            if len(final_model.training_losses) < 200:
                del final_model
            else:
                break

        # plot_suffix = '{}'.format(fold_index)
        # final_model.get_plot(plot_suffix=plot_suffix)
        prediction_train = final_model.predict(X, y)
        prediction_test = final_model.predict(test_data[0], test_data[1])
        fold_train_losses = final_model.training_losses

        violations_train_prediction = count_violations(X, prediction_train[0])
        violations_test_prediction = count_violations(test_data[0], prediction_test[0])

        violations_train_data = count_violations(X, y)
        violations_test_data = count_violations(test_data[0], test_data[1])

        correct_prediction_train = [1 if y[i] == prediction_train[0][i] else 0 for i in range(X.shape[0])]
        correct_prediction_train = [1 if X['group_The Beatles'][i] == 1 and prediction_train[0][i] in ['Pop', 'Rock']
                                    else correct_prediction_train[i] for i in range(X.shape[0])]

        corrected_train_accuracy = sum(correct_prediction_train) / X.shape[0]

        correct_prediction_test = [1 if test_data[1][i] == prediction_test[0][i] else 0 for i in
                                   range(test_data[0].shape[0])]
        correct_prediction_test = [1 if test_data[0]['group_The Beatles'][i] == 1 and prediction_test[0][i] in
                                        ['Pop', 'Rock'] else correct_prediction_test[i]
                                   for i in range(test_data[0].shape[0])]

        corrected_test_accuracy = sum(correct_prediction_test) / test_data[0].shape[0]

        counter_examples = find_counter_example_2(X, parameters=final_model.W_learnt)
        output = output.append([[fold_index, M, K, learning_rate, batch_size, epoch, final_model.runtime,
                                 final_model.W_learnt, violations_train_data, corrected_train_accuracy,
                                 violations_train_prediction, violations_test_data, corrected_test_accuracy,
                                 violations_test_prediction, len(counter_examples), len(final_model.training_losses)]])

        output.columns = ['fold', 'margin/alpha', 'knowledge', 'learning_rate', 'batch_size', 'epochs', 'runtime',
                          'weights', 'violations_train_data', 'accuracy_train_corrected', 'violations_train_prediction',
                          'violations_test_data', 'accuracy_test_corrected', 'violations_test_prediction',
                          'counter_examples_found', 'num_iterations']

        return output, CV_log, fold_train_losses, val_evaluation


def single_baseline_run(X, y, test_data=None, params=None, fold_index=None):
    output = pd.DataFrame()
    alphas = ['PP', 0, 1, 2, 5, 10, 50, 100]

    loss_types = ['sbr', 'sl']
    K = params['K'][0]

    for l in loss_types:
        for alpha in alphas:
            model = MusicGenreClassificationRegularization(loss_type=l)
            start = time.time()
            if alpha == 'PP':
                model.learn(X, y, alpha=0)
                y_train_b = model.predict(X)[0]
                y_train_b = [[y_train_b[j][i] for j in range(len(np.unique(y)))] for i in range(X.shape[0])]
                y_train_b = [np.unique(y)[3:][y_train_b[i][3:].index(max(y_train_b[i][3:]))]
                             if X['group_The Beatles'][i] == 1 else np.unique(y)[y_train_b[i].index(max(y_train_b[i]))]
                             for i in range(X.shape[0])]

                y_test_b = model.predict(test_data[0])[0]
                y_test_b = [[y_test_b[j][i] for j in range(len(np.unique(y)))] for i in range(test_data[0].shape[0])]
                y_test_b = [np.unique(y)[3:][y_test_b[i][3:].index(max(y_test_b[i][3:]))] if
                            test_data[0]['group_The Beatles'][i] == 1
                            else np.unique(y)[y_test_b[i].index(max(y_test_b[i]))]
                            for i in range(test_data[0].shape[0])]
            else:
                model.learn(X, y, alpha=alpha)
                y_train_b = model.predict(X)[1]
                y_test_b = model.predict(test_data[0])[1]

            runtime = time.time() - start
            violations_train_prediction = count_violations(X, y_train_b)
            violations_test_prediction = count_violations(test_data[0], y_test_b)

            violations_train_data = count_violations(X, y)
            violations_test_data = count_violations(test_data[0], test_data[1])

            correct_prediction_train = [1 if y[i] == y_train_b[i] else 0 for i in range(X.shape[0])]
            correct_prediction_train = [1 if X['group_The Beatles'][i] == 1 and y_train_b[i] in ['Pop', 'Rock']
                                        else correct_prediction_train[i] for i in range(X.shape[0])]

            corrected_train_accuracy = sum(correct_prediction_train) / X.shape[0]

            correct_prediction_test = [1 if test_data[1][i] == y_test_b[i] else 0 for i in range(test_data[0].shape[0])]
            correct_prediction_test = [1 if test_data[0]['group_The Beatles'][i] == 1 and y_test_b[i] in
                                            ['Pop', 'Rock'] else correct_prediction_test[i]
                                       for i in range(test_data[0].shape[0])]

            corrected_test_accuracy = sum(correct_prediction_test) / test_data[0].shape[0]

            weights_learned = [list(model.beta[i * (X.shape[1] + 1):(i + 1) * (X.shape[1] + 1)])
                               for i in range(len(np.unique(y)))]

            if alpha == 'PP':
                counter_examples = []
            else:
                counter_examples = find_counter_example_2(X, parameters=weights_learned)
            output = output.append([[fold_index, l, alpha, K, '-', '-', '-', runtime,
                                     weights_learned, violations_train_data, corrected_train_accuracy,
                                     violations_train_prediction, violations_test_data, corrected_test_accuracy,
                                     violations_test_prediction, len(counter_examples)]])
            model.reset()

    output.columns = ['fold', 'loss_type', 'margin/alpha', 'knowledge', 'learning_rate', 'batch_size', 'epochs',
                      'runtime', 'weights', 'violations_train_data', 'accuracy_train_corrected',
                      'violations_train_prediction', 'violations_test_data', 'accuracy_test_corrected',
                      'violations_test_prediction', 'counter_examples_found']

    return output


def single_baseline_run_2(X, y, test_data=None, params=None, fold_index=None):
    output = pd.DataFrame()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=random.randint(0, 100))
    X_train.reset_index(inplace=True, drop=True)
    X_val.reset_index(inplace=True, drop=True)
    alphas = ['PP', 0, 1, 2, 5, 10, 50, 100]

    loss_types = ['sbr', 'sl']
    K = params['K'][0]

    for l in loss_types:
        for alpha in alphas:
            model = MusicGenreClassificationRegularization(loss_type=l)
            start = time.time()
            if alpha == 'PP':
                model.learn(X_train, y_train, alpha=0)
                y_train_b = model.predict(X_train)[0]
                y_train_b = [[y_train_b[j][i] for j in range(len(np.unique(y_train)))] for i in range(X_train.shape[0])]
                y_train_b = [np.unique(y_train)[3:][y_train_b[i][3:].index(max(y_train_b[i][3:]))]
                             if X_train['group_The Beatles'][i] == 1 else np.unique(y_train)[
                    y_train_b[i].index(max(y_train_b[i]))]
                             for i in range(X_train.shape[0])]

                y_val_b = model.predict(X_val)[0]
                y_val_b = [[y_val_b[j][i] for j in range(len(np.unique(y_val)))] for i in range(X_val.shape[0])]
                y_val_b = [np.unique(y_val)[3:][y_val_b[i][3:].index(max(y_val_b[i][3:]))]
                           if X_val['group_The Beatles'][i] == 1 else np.unique(y_val)[
                    y_val_b[i].index(max(y_val_b[i]))]
                           for i in range(X_val.shape[0])]

                y_test_b = model.predict(test_data[0])[0]
                y_test_b = [[y_test_b[j][i] for j in range(len(np.unique(y)))] for i in range(test_data[0].shape[0])]
                y_test_b = [np.unique(y)[3:][y_test_b[i][3:].index(max(y_test_b[i][3:]))] if
                            test_data[0]['group_The Beatles'][i] == 1
                            else np.unique(y)[y_test_b[i].index(max(y_test_b[i]))]
                            for i in range(test_data[0].shape[0])]
            else:
                model.learn(X_train, y_train, alpha=alpha)
                y_train_b = model.predict(X_train)[1]
                y_val_b = model.predict(X_val)[1]
                y_test_b = model.predict(test_data[0])[1]

            runtime = time.time() - start
            violations_train_prediction = count_violations(X_train, y_train_b)
            violations_val_prediction = count_violations(X_val, y_val_b)
            violations_test_prediction = count_violations(test_data[0], y_test_b)

            violations_train_data = count_violations(X_train, y_train)
            violations_val_data = count_violations(X_val, y_val)
            violations_test_data = count_violations(test_data[0], test_data[1])

            correct_prediction_train = [1 if y_train[i] == y_train_b[i] else 0 for i in range(X_train.shape[0])]
            correct_prediction_train = [1 if X_train['group_The Beatles'][i] == 1 and y_train_b[i] in ['Pop', 'Rock']
                                        else correct_prediction_train[i] for i in range(X_train.shape[0])]

            corrected_train_accuracy = sum(correct_prediction_train) / X_train.shape[0]

            correct_prediction_val = [1 if y_val[i] == y_val_b[i] else 0 for i in range(X_val.shape[0])]
            correct_prediction_val = [1 if X_val['group_The Beatles'][i] == 1 and y_val_b[i] in ['Pop', 'Rock']
                                      else correct_prediction_val[i] for i in range(X_val.shape[0])]

            corrected_val_accuracy = sum(correct_prediction_val) / X_val.shape[0]

            correct_prediction_test = [1 if test_data[1][i] == y_test_b[i] else 0 for i in range(test_data[0].shape[0])]
            correct_prediction_test = [1 if test_data[0]['group_The Beatles'][i] == 1 and y_test_b[i] in
                                            ['Pop', 'Rock'] else correct_prediction_test[i]
                                       for i in range(test_data[0].shape[0])]

            corrected_test_accuracy = sum(correct_prediction_test) / test_data[0].shape[0]

            weights_learned = [list(model.beta[i * (X_train.shape[1] + 1):(i + 1) * (X_train.shape[1] + 1)])
                               for i in range(len(np.unique(y_train)))]

            if alpha == 'PP':
                counter_examples = []
            else:
                counter_examples = find_counter_example_2(X, parameters=weights_learned)
            output = output.append([[fold_index, l, alpha, K, '-', '-', '-', runtime,
                                     weights_learned, violations_train_data, corrected_train_accuracy,
                                     violations_train_prediction, violations_val_data, corrected_val_accuracy,
                                     violations_val_prediction, violations_test_data, corrected_test_accuracy,
                                     violations_test_prediction, len(counter_examples)]])
            model.reset()

    output.columns = ['fold', 'loss_type', 'margin/alpha', 'knowledge', 'learning_rate', 'batch_size', 'epochs',
                      'runtime', 'weights', 'violations_train_data', 'accuracy_train_corrected',
                      'violations_train_prediction', 'violations_val_data', 'accuracy_val_corrected',
                      'violations_val_prediction', 'violations_test_data', 'accuracy_test_corrected',
                      'violations_test_prediction', 'counter_examples_found']

    return output


if __name__ == "__main__":
    start = time.time()
    dataset = 'music_data_24122020.csv'
    data = pd.read_csv(dataset, header=0).iloc[:, 1:]

    folder_name = sys.argv[6]
    if len(folder_name.split('_')) > 1:
        experiment_number = eval(folder_name.split('_')[0])
    else:
        experiment_number = eval(folder_name)
    random.seed(experiment_number ** 3)

    data = data.sample(frac=1, random_state=random.randint(1, 1000)).reset_index(drop=True)
    instances_to_change = [i for i in range(data.shape[0]) if data['group'][i] == 'The Beatles']
    print(len(instances_to_change))
    instances_to_change = random.sample(instances_to_change, 60)

    for i in instances_to_change:
        data['Music Style'][i] = random.choice(['Metal', 'Electronic', 'Classical'])

    X_original = data.iloc[:, :-1]
    y = list(data.iloc[:, -1])

    categorical_features = [f for f in list(X_original.columns) if f not in set(X_original._get_numeric_data().columns)]
    numerical_features = list(set(X_original._get_numeric_data().columns))
    scaler = MinMaxScaler()

    X_original[numerical_features] = scaler.fit_transform(X_original[numerical_features])
    X = pd.get_dummies(X_original, columns=categorical_features, drop_first=False)

    print(X.columns)
    print(X.head())
    outer_folds = 5
    kf = KFold(n_splits=outer_folds, shuffle=True, random_state=random.randint(0, 1000))
    split_data = list(kf.split(X))

    params = {'M': eval(sys.argv[1]),
              'K': [0 if sys.argv[2] == '1' else None],
              'epoch': eval(sys.argv[3]),
              'learning_rate': eval(sys.argv[4]),
              'batch_size': eval(sys.argv[5])}

    print('all parameters:', params)
    f = 1
    all_output = pd.DataFrame()
    all_CV_logs = pd.DataFrame()
    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []
    list_5 = []
    for train_index, test_index in split_data:
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = [y[i] for i in train_index], \
                          [y[i] for i in test_index]
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        list_1.append(X_train)
        list_2.append(y_train)
        list_3.append([X_test, y_test])
        list_4.append(params)
        list_5.append(f)
        f = f + 1

    if not os.path.exists('{}'.format(folder_name)):
        os.makedirs('{}'.format(folder_name))
    
    do_baseline = True
    if do_baseline:
        baseline_results = list(futures.map(single_baseline_run_2, list_1, list_2, list_3, list_4, list_5))
        all_baseline_output = pd.DataFrame()
        for out in baseline_results:
            all_baseline_output = all_baseline_output.append(out)
        all_baseline_output.to_csv(folder_name + '/all_baseline_outputs.csv')

    # SaDe
    results_sade_l = list(futures.map(single_model_run, list_1, list_2, list_3, list_4, list_5))
    all_output_sade_l = pd.DataFrame()
    all_CV_logs_sade_l = pd.DataFrame()
    all_CV_eval_sade_l = pd.DataFrame()
    all_fold_test_losses_sade_l = []
    print(results_sade_l[0])

    for out in results_sade_l:
        all_output_sade_l = all_output_sade_l.append(out[0])
        all_CV_logs_sade_l = all_CV_logs_sade_l.append(out[1])
        all_fold_test_losses_sade_l.append(out[2])
        all_CV_eval_sade_l = all_CV_eval_sade_l.append(out[3])

    all_output_sade_l.to_csv(folder_name + '/all_outputs_sade_l.csv')
    all_CV_logs_sade_l.to_csv(folder_name + '/all_CV_logs_sade_l.csv')
    all_CV_eval_sade_l.to_csv(folder_name + '/all_CV_eval_sade_l.csv')

    with open(folder_name + '/all_test_losses_sade_l.pkl', 'wb') as f:
        pickle.dump(all_fold_test_losses_sade_l, f)

    print('total time of the run:', time.time() - start)
