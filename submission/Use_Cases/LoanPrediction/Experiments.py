import sys
from scoop import futures
from FeatureConstraintsRegularization import *
import pickle
from Supplements import *
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
            val_evaluation, best_parameter, CV_log = \
                cross_validate_parameter_selection(X, y, params=params)

            M, K, learning_rate, epoch, batch_size, validation_loss = best_parameter
            val_evaluation['outer_fold'] = fold_index
        while True:
            final_model = LinearClassificationLearnerLoan()
            final_model.learn(X, y, M=M, K=K, validation_set=None,
                              batch_size=batch_size, epochs=epoch, learning_rate=learning_rate,
                              learner='fumalik', step_size=1, waiting_period=5, early_stopping=True)
            if len(final_model.training_losses) < 100:
                del final_model
            else:
                f = open("log.txt", "a+")
                f.write('selected a model for fold: {}\n'.format(fold_index))
                f.close()
                break

        # plot_suffix = '{}'.format(fold_index)
        # final_model.get_plot(plot_suffix=plot_suffix)
        prediction_train = final_model.predict(X, y)
        fold_train_losses = final_model.training_losses

        correct_prediction = 0
        for i in range(X.shape[0]):
            if X['Credit_History'][i] == 0 and X['ApplicantIncome'][i] <= K[0]:
                if prediction_train[0][i] == 0:
                    correct_prediction += 1
            else:
                if y[i] == prediction_train[0][i]:
                    correct_prediction += 1

        accuracy_train_z3_corrected = correct_prediction / X.shape[0]
        accuracy_train_z3_modified = calculate_modified_accuracy(X, y, prediction_train[0], K)

        prediction_test = final_model.predict(test_data[0], test_data[1])

        correct_prediction = 0
        for i in range(test_data[0].shape[0]):
            if test_data[0]['Credit_History'][i] == 0 and test_data[0]['ApplicantIncome'][i] <= K[0]:
                if prediction_test[0][i] == 0:
                    correct_prediction += 1
            else:
                if test_data[1][i] == prediction_test[0][i]:
                    correct_prediction += 1

        accuracy_test_z3_corrected = correct_prediction / test_data[0].shape[0]
        accuracy_test_z3_modified = calculate_modified_accuracy(test_data[0], test_data[1], prediction_test[0], K)

        violations_train_z3 = count_violations(X, prediction_train[0], K[0])
        violations_test_z3 = count_violations(test_data[0], prediction_test[0], K[0])

        counter_example_count_z3 = find_counter_example(X, final_model.W_learnt, K=K)
        output = output.append(
            [[fold_index, M, K, learning_rate, batch_size, epoch, final_model.runtime, final_model.W_learnt,
              count_violations(X, y, K[0])[0], accuracy_train_z3_corrected, accuracy_train_z3_modified,
              violations_train_z3[0], count_violations(test_data[0], test_data[1], K[0])[0],
              accuracy_test_z3_corrected, accuracy_test_z3_modified, violations_test_z3[0],
              len(counter_example_count_z3), len(final_model.training_losses)]])

        output.columns = ['fold', 'margin/alpha', 'knowledge', 'learning_rate', 'batch_size', 'epochs', 'runtime',
                          'weights', 'violations_train_data', 'accuracy_train_corrected', 'accuracy_train_modified',
                          'violations_train_prediction', 'violations_test_data', 'accuracy_test_corrected',
                          'accuracy_test_modified', 'violations_test_prediction',  'counter_examples_found',
                          'num_iteration']
        return output, CV_log, fold_train_losses, val_evaluation


def single_baseline_run_2(X, y, test_data=None, params=None, fold_index=None):
    output = pd.DataFrame()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=random.randint(0, 1000))
    X_train.reset_index(inplace=True, drop=True)
    X_val.reset_index(inplace=True, drop=True)

    print(X_train.shape, X_val.shape, test_data[0].shape)
    loss_types = ['sbr', 'sl']
    alphas = ['PP', 0, 0.1, 0.5, 1, 2, 5, 10, 100, 500, 1000, 10000]
    K = params['K'][0]

    for l in loss_types:
        for alpha in alphas:
            model = LoanPredictionRegularization(loss_type=l)
            if alpha == 'PP':
                model.learn(X_train, y_train, alpha=0, K=K)
                y_train_b = model.predict(X_train)[1]
                for i in range(X_train.shape[0]):
                    if X_train['Credit_History'][i] == 0 and X_train['ApplicantIncome'][i] <= K[0]:
                        y_train_b[i] = 0

                y_val_b = model.predict(X_val)[1]
                for i in range(X_val.shape[0]):
                    if X_val['Credit_History'][i] == 0 and X_val['ApplicantIncome'][i] <= K[0]:
                        y_val_b[i] = 0

                y_test_b = model.predict(test_data[0])[1]
                for i in range(test_data[0].shape[0]):
                    if test_data[0]['Credit_History'][i] == 0 and test_data[0]['ApplicantIncome'][i] <= K[0]:
                        y_test_b[i] = 0
            else:
                model.learn(X_train, y_train, alpha=alpha, K=K)
                y_train_b = model.predict(X_train)[1]
                y_val_b = model.predict(X_val)[1]
                y_test_b = model.predict(test_data[0])[1]

            correct_prediction = 0
            for i in range(test_data[0].shape[0]):
                if test_data[0]['Credit_History'][i] == 0 and test_data[0]['ApplicantIncome'][i] <= K[0]:
                    if y_test_b[i] == 0:
                        correct_prediction += 1
                else:
                    if test_data[1][i] == y_test_b[i]:
                        correct_prediction += 1

            accuracy_test_b_corrected = correct_prediction / test_data[0].shape[0]
            accuracy_test_b_modified = calculate_modified_accuracy(test_data[0], test_data[1], y_test_b, K)

            correct_prediction = 0
            for i in range(X_train.shape[0]):
                if X_train['Credit_History'][i] == 0 and X_train['ApplicantIncome'][i] <= K[0]:
                    if y_train_b[i] == 0:
                        correct_prediction += 1
                else:
                    if y_train[i] == y_train_b[i]:
                        correct_prediction += 1

            accuracy_train_b_corrected = correct_prediction / X_train.shape[0]
            accuracy_train_b_modified = calculate_modified_accuracy(X_train, y_train, y_train_b, K)

            correct_prediction = 0
            for i in range(X_val.shape[0]):
                if X_val['Credit_History'][i] == 0 and X_val['ApplicantIncome'][i] <= K[0]:
                    if y_val_b[i] == 0:
                        correct_prediction += 1
                else:
                    if y_val[i] == y_val_b[i]:
                        correct_prediction += 1

            accuracy_val_b_corrected = correct_prediction / X_val.shape[0]
            accuracy_val_b_modified = calculate_modified_accuracy(X_val, y_val, y_val_b, K)

            violations_train_b = count_violations(X_train, y_train_b, K[0])[0]
            violations_val_b = count_violations(X_val, y_val_b, K[0])[0]
            violations_test_b = count_violations(test_data[0], y_test_b, K[0])[0]

            if alpha == 'PP':
                counter_example_count_b = []
            else:
                counter_example_count_b = find_counter_example(X, model.beta, K=K)
            if len(counter_example_count_b) > 0:
                print(counter_example_count_b[0])

            output = output.append(
                [[fold_index, l, alpha, '-', '-', '-', '-', model.runtime, model.beta,
                  count_violations(X_train, y_train, K[0])[0], accuracy_train_b_corrected, accuracy_train_b_modified,
                  violations_train_b, count_violations(X_val, y_val, K[0])[0], accuracy_val_b_corrected,
                  accuracy_val_b_modified, violations_val_b, count_violations(test_data[0], test_data[1], K[0])[0],
                  accuracy_test_b_corrected, accuracy_test_b_modified, violations_test_b,
                  len(counter_example_count_b)]])

            model.reset()

    output.columns = ['fold', 'loss_type', 'margin/alpha', 'knowledge', 'learning_rate', 'batch_size', 'epochs',
                      'runtime', 'weights', 'violations_train_data', 'accuracy_train_corrected',
                      'accuracy_train_modified', 'violations_train_prediction', 'violations_val_data',
                      'accuracy_val_corrected', 'accuracy_val_modified',  'violations_val_prediction',
                      'violations_test_data', 'accuracy_test_corrected', 'accuracy_test_modified',
                      'violations_test_prediction', 'counter_examples_found']

    return output


if __name__ == "__main__":
    start = time.time()
    dataset = 'loan_data_set.csv'
    data = pd.read_csv(dataset, header=0)
    experiment_number = eval(sys.argv[6])
    
    random.seed(experiment_number ** 3)
    
    processed_output = data_processing(data)
    data = processed_output[0]
    X = data.iloc[:, 0:data.shape[1] - 1]
    y_original = list(data.iloc[:, -1])
    scaled_income = processed_output[1].scale_[0] * (5000 - processed_output[1].data_min_[0])

    X, y = data_manipulation(X, y_original)
    f = open("CV_log.txt", "a+")
    f.write('violations in full data: {}\n'.format(count_violations(X, y, scaled_income)[0]))
    f.close()
    outer_folds = 5
    
    kf = KFold(n_splits=outer_folds, shuffle=True, random_state=random.randint(0, 1000))
    split_data = list(kf.split(X))

    params = {'M': eval(sys.argv[1]),
              'K': [[scaled_income, 0] if sys.argv[2] == '1' else [scaled_income, None]],
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

    do_baseline = True
    
    if not os.path.exists('{}'.format(str(experiment_number))):
        os.makedirs('{}'.format(str(experiment_number)))

    if do_baseline:
        all_baseline_output = pd.DataFrame()
        baseline_results = list(futures.map(single_baseline_run_2, list_1, list_2, list_3, list_4, list_5))
        for out in baseline_results:
            all_baseline_output = all_baseline_output.append(out)
        print(all_baseline_output)
        all_baseline_output.to_csv(str(experiment_number) + '/all_baseline_outputs.csv')

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

    all_output_sade_l.to_csv(str(experiment_number) + '/all_outputs_sade_l.csv')
    all_CV_logs_sade_l.to_csv(str(experiment_number) + '/all_CV_logs_sade_l.csv')
    all_CV_eval_sade_l.to_csv(str(experiment_number) + '/all_CV_eval_sade_l.csv')

    with open(str(experiment_number) + '/all_test_losses_sade_l.pkl', 'wb') as f:
        pickle.dump(all_fold_test_losses_sade_l, f)

    print('total time of the run:', time.time() - start)
