from Supplements import *
from CrossValidationMultiTargetRegression import *
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from BoundedMultiTargetRegressionRegularization import *
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
                cross_validate_parameter_selection(X, y, params=params)
            # cross_validate_parameter_selection(BoundedMultiTargetLearner,
            #                                    X, y, params=params)
            val_evaluation['outer_fold'] = fold_index
            M, K, learning_rate, epoch, batch_size, validation_loss = best_parameter

        while True:
            final_model = BoundedMultiTargetLearner()
            final_model.learn(X, y, M=M, K=K, validation_set=None,
                              batch_size=batch_size, epochs=epoch, learning_rate=learning_rate, learner='fumalik', step_size=1, waiting_period=5, early_stopping=False)
            if len(final_model.training_losses) < 100:
                del final_model
            else:
                break

        # plot_suffix = '{}'.format(fold_index)
        # final_model.get_plot(plot_suffix=plot_suffix)

        fold_train_losses = final_model.training_losses
        prediction_train = final_model.predict(X, y)
        prediction_train_instances = [sum(i) for i in zip(*prediction_train[0])]
        y_train_instances = [sum(i) for i in zip(*y)]

        custom_metric_train_1 = '-'

        custom_metric_train_2 = '-'

        prediction_test = final_model.predict(test_data[0], test_data[1])
        prediction_test_instances = [sum(i) for i in zip(*prediction_test[0])]
        y_test_instances = [sum(i) for i in zip(*test_data[1])]

        custom_metric_test_1 = '-'

        custom_metric_test_2 = '-'

        violations_train = count_violations(X, prediction_train[0], K=K)[0]
        violations_test_all = count_violations(test_data[0], prediction_test[0], K=K)
        violations_test = violations_test_all[0]

        counter_examples = len(find_counter_example(X, final_model.W_learnt, K=K))

        output = output.append([[fold_index, M, K, learning_rate, batch_size, epoch, final_model.runtime,
                                 final_model.W_learnt, count_violations(X, y, K=K)[0], sum(prediction_train[1]),
                                 custom_metric_train_1, custom_metric_train_2, violations_train,
                                 count_violations(test_data[0], test_data[1], K=K)[0], sum(prediction_test[1]),
                                 custom_metric_test_1, custom_metric_test_2, violations_test, counter_examples]])

        output.columns = ['fold', 'margin/alpha', 'knowledge', 'learning_rate', 'batch_size', 'epochs', 'runtime',
                          'weights', 'violations_train_data', 'total_mse_train', 'custom_metric_train_1',
                          'custom_metric_train_2', 'violations_train_prediction', 'violations_test_data',
                          'total_mse_test', 'custom_metric_test_1', 'custom_metric_test_2',
                          'violations_test_prediction', 'counter_examples_found']

        return output, CV_log, fold_training_losses, val_evaluation, violations_test_all[1]


def single_baseline_run_2(X, y, test_data=None, params=None, fold_index=None):
    output = pd.DataFrame()
    X_train, X_val, y0_train, y0_val, y1_train, y1_val, y2_train, y2_val, y3_train, y3_val, y4_train, y4_val = \
        train_test_split(X, y[0], y[1], y[2], y[3], y[4], test_size=0.25, random_state=random.randint(0, 1000))
    y_train = [y0_train, y1_train, y2_train, y3_train, y4_train]
    y_val = [y0_val, y1_val, y2_val, y3_val, y4_val]
    X_train.reset_index(inplace=True, drop=True)
    X_val.reset_index(inplace=True, drop=True)

    alphas = [0, 1, 2, 5, 10, 20, 50, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]
    model = BoundedPredictionRegularization()
    K = params['K'][0]
    y_train_instances = [sum(i) for i in zip(*y_train)]
    y_val_instances = [sum(i) for i in zip(*y_val)]
    y_test_instances = [sum(i) for i in zip(*test_data[1])]

    for alpha in alphas:
        baseline_start = time.time()
        if alpha == 'PP':
            model.learn(X_train, y_train, alpha=0, K=K)
            prediction_train = model.predict(X_train, y_train)
            prediction_train_instances = [sum(i) for i in zip(*prediction_train[0])]
            correction_train = [
                max(prediction_train_instances[i] - ((X_train['Total Household Income'][i] - K[0]) / K[1]), 0)
                for i in range(X_train.shape[0])]
            prediction_train = list(prediction_train)
            prediction_train[0] = [
                [prediction_train[0][j][i] * (1 - correction_train[i] / prediction_train_instances[i])
                 for i in range(X_train.shape[0])] for j in range(len(y_train))]
            prediction_train[1] = [mean_squared_error(y_train[j], prediction_train[0][j]) for j in range(len(y_train))]
            prediction_train_instances = [sum(i) for i in zip(*prediction_train[0])]

            prediction_val = model.predict(X_val, y_val)
            prediction_val_instances = [sum(i) for i in zip(*prediction_val[0])]
            correction_val = [max(prediction_val_instances[i] - ((X_val['Total Household Income'][i] - K[0]) / K[1]), 0)
                              for i in range(X_val.shape[0])]
            prediction_val = list(prediction_val)
            prediction_val[0] = [
                [prediction_val[0][j][i] * (1 - correction_val[i] / prediction_val_instances[i])
                 for i in range(X_val.shape[0])] for j in range(len(y_val))]
            prediction_val[1] = [mean_squared_error(y_val[j], prediction_val[0][j]) for j in range(len(y_val))]
            prediction_val_instances = [sum(i) for i in zip(*prediction_val[0])]

            prediction_test = model.predict(test_data[0], test_data[1])
            prediction_test_instances = [sum(i) for i in zip(*prediction_test[0])]
            correction_test = [max(prediction_test_instances[i] -
                                   ((test_data[0]['Total Household Income'][i] - K[0]) / K[1]), 0)
                               for i in range(test_data[0].shape[0])]
            prediction_test = list(prediction_test)

            prediction_test[0] = [[prediction_test[0][j][i] * (1 - correction_test[i] / prediction_test_instances[i])
                                   for i in range(test_data[0].shape[0])] for j in range(len(test_data[1]))]
            # prediction_test[0] = [[prediction_test[0][j][i] - correction_test[i]/5
            # for i in range(test_data[0].shape[0])] for j in range(len(test_data[1]))]
            prediction_test[1] = [mean_squared_error(test_data[1][j], prediction_test[0][j]) for j in
                                  range(len(test_data[1]))]
            prediction_test_instances = [sum(i) for i in zip(*prediction_test[0])]
        else:
            model.learn(X_train, y_train, alpha=alpha, K=K)
            prediction_train = model.predict(X_train, y_train)
            prediction_train_instances = [sum(i) for i in zip(*prediction_train[0])]
            prediction_val = model.predict(X_val, y_val)
            prediction_val_instances = [sum(i) for i in zip(*prediction_val[0])]
            prediction_test = model.predict(test_data[0], test_data[1])
            prediction_test_instances = [sum(i) for i in zip(*prediction_test[0])]

        runtime = time.time() - baseline_start

        custom_metric_train_1 = '-'

        custom_metric_train_2 = '-'

        custom_metric_val_1 = '-'

        custom_metric_val_2 = '-'

        custom_metric_test_1 = '-'

        custom_metric_test_2 = '-'

        violations_train = count_violations(X_train, prediction_train[0], K=K)[0]
        violations_val = count_violations(X_val, prediction_val[0], K=K)[0]
        violations_test = count_violations(test_data[0], prediction_test[0], K=K)[0]

        weights_learned = [list(model.beta[i * (X_train.shape[1] + 1):(i + 1) * (X_train.shape[1] + 1)])
                           for i in range(len(y_train))]

        if alpha == 'PP':
            counter_examples = 0
        else:
            counter_examples = len(find_counter_example(X, weights_learned, K=K))

        output = output.append([[fold_index, alpha, K, '-', '-', '-', runtime,
                                 weights_learned, count_violations(X_train, y_train, K=K)[0], sum(prediction_train[1]),
                                 custom_metric_train_1, custom_metric_train_2, violations_train,
                                 count_violations(X_val, y_val, K=K)[0], sum(prediction_val[1]),
                                 custom_metric_val_1, custom_metric_val_2, violations_val,
                                 count_violations(test_data[0], test_data[1], K=K)[0], sum(prediction_test[1]),
                                 custom_metric_test_1, custom_metric_test_2, violations_test, counter_examples]])

        model.reset()

    output.columns = ['fold', 'margin/alpha', 'knowledge', 'learning_rate', 'batch_size', 'epochs', 'runtime',
                      'weights', 'violations_train_data', 'total_mse_train', 'custom_metric_train_1',
                      'custom_metric_train_2', 'violations_train_prediction',
                      'violations_val_data', 'total_mse_val', 'custom_metric_val_1',
                      'custom_metric_val_2', 'violations_val_prediction', 'violations_test_data',
                      'total_mse_test', 'custom_metric_test_1', 'custom_metric_test_2', 'violations_test_prediction',
                      'counter_examples_found']

    return output


if __name__ == "__main__":
    start = time.time()
    data_path = 'datasets/family_income_24122020.csv'
    data = pd.read_csv(data_path, header=0).iloc[:, 1:]
    data = data.iloc[:1000, :]
    print(data.head())

    folder_name = sys.argv[6]
    if len(folder_name.split('_')) > 1:
        experiment_number = eval(folder_name.split('_')[0])
    else:
        experiment_number = eval(folder_name)
    random.seed(experiment_number ** 3)

    n_features = data.shape[1] - 1
    n_instances = len(data)
    y = data.iloc[:, -5:]
    X = data.iloc[:, 0:data.shape[1] - 5]
    X['Household Head Sex'] = [1 if x == 'Male' else 0 for x in X['Household Head Sex']]

    scaler = MinMaxScaler()
    # A = [x for x in X.iloc[:, -1]]

    X[X.columns] = scaler.fit_transform(X[X.columns])
    for i in range(X.shape[1]):
        print(X.columns[i], min(X.iloc[:, i]), max(X.iloc[:, i]))
    y = [list(y.iloc[:, i]) for i in range(y.shape[1])]

    outer_folds = 5
    kf = KFold(n_splits=outer_folds, shuffle=True, random_state=random.randint(0, 1000))
    split_data = list(kf.split(X))

    params = {'M': eval(sys.argv[1]),
              'K': [[scaler.min_[-1], scaler.scale_[-1], 0] if sys.argv[2] == '1' else None],
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
        y_train, y_test = [[v[i] for i in train_index] for v in y], \
                          [[v[i] for i in test_index] for v in y]
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        list_1.append(X_train)
        list_2.append(y_train)
        list_3.append([X_test, y_test])
        list_4.append(params)
        list_5.append(f)

        f = f + 1

    do_baseline = False
    
    if not os.path.exists('{}'.format(folder_name)):
        os.makedirs('{}'.format(folder_name))
    
    if do_baseline:
        baseline_results = list(futures.map(single_baseline_run_2, list_1, list_2, list_3, list_4, list_5))
        all_baseline_output = pd.DataFrame()
        for out in baseline_results:
            all_baseline_output = all_baseline_output.append(out)

        print(all_baseline_output)
        all_baseline_output.to_csv(folder_name + '/all_baseline_outputs.csv')

    # SaDe
    results_sade_l = list(futures.map(single_model_run, list_1, list_2, list_3, list_4, list_5))
    all_output_sade_l = pd.DataFrame()
    all_CV_logs_sade_l = pd.DataFrame()
    all_CV_eval_sade_l = pd.DataFrame()
    all_fold_test_losses_sade_l = []

    i = 1
    for out in results_sade_l:
        all_output_sade_l = all_output_sade_l.append(out[0])
        all_CV_logs_sade_l = all_CV_logs_sade_l.append(out[1])
        all_fold_test_losses_sade_l.append(out[2])
        all_CV_eval_sade_l = all_CV_eval_sade_l.append(out[3])
        if out[4].shape[0] > 0:
            out[4].to_csv(folder_name + '/violating_instances_{}.csv'.format(i))
        i += 1

    all_output_sade_l.to_csv(folder_name + '/all_outputs_sade_l.csv')
    all_CV_logs_sade_l.to_csv(folder_name + '/all_CV_logs_sade_l.csv')
    all_CV_eval_sade_l.to_csv(folder_name + '/all_CV_eval_sade_l.csv')

    with open(folder_name + '/all_test_losses_sade_l.pkl', 'wb') as f:
        pickle.dump(all_fold_test_losses_sade_l, f)

    print('total time of the run:', time.time() - start)
