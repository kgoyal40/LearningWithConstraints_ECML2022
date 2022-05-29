import sys
from sklearn.model_selection import KFold
from scoop import futures
from BoundedMultiTargetRegression import *


def single_run(X, y, M, K, test_set, batch_size, epoch, learning_rate, fold):
    out = pd.DataFrame()

    model = BoundedMultiTargetLearner()
    model.learn(X=X, y=y, M=M, K=K, validation_set=None, batch_size=batch_size,
                epochs=epoch, learning_rate=learning_rate, learner='fumalik')

    if len(model.training_losses) > 100:
        val_loss = model.calculate_gradient(test_set[0], test_set[1], model.W_learnt)[1]
        # f = open("CV_log.txt", "a+")
        # f.write('cv_log_entry \n')
        # f.close()
        out = out.append([[str(fold), str(M), str(K), str(learning_rate), str(batch_size), str(epoch),
                           val_loss, model.runtime,
                           len(model.training_losses)]])

    return out


def cross_validate_parameter_selection(X, y, params=None, n_folds=5):
    if params is None:
        print('please provide parameter values')
        sys.exit(0)

    all_models = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random.randint(0, 1000))
    split_data = list(kf.split(X))
    CV_log = pd.DataFrame()
    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []
    list_5 = []
    list_6 = []
    list_7 = []
    list_8 = []
    list_9 = []

    for M in params['M']:
        for K in params['K']:
            for learning_rate in params['learning_rate']:
                for epoch in params['epoch']:
                    for batch_size in params['batch_size']:
                        fold = 1
                        for train_index, val_index in split_data:
                            X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
                            y_train, y_val = [[v[i] for i in train_index] for v in y], \
                                             [[v[i] for i in val_index] for v in y]
                            X_train.reset_index(drop=True, inplace=True)
                            X_val.reset_index(drop=True, inplace=True)
                            list_1.append(X_train)
                            list_2.append(y_train)
                            list_3.append(M)
                            list_4.append(K)
                            list_5.append([X_val, y_val])
                            list_6.append(batch_size)
                            list_7.append(epoch)
                            list_8.append(learning_rate)
                            list_9.append(fold)
                            fold = fold + 1

    results = list(futures.map(single_run, list_1, list_2, list_3, list_4,
                               list_5, list_6, list_7, list_8, list_9))

    for out in results:
        CV_log = CV_log.append(out)

    CV_log.columns = ['fold', 'margin', 'knowledge', 'learning_rate', 'batch_size', 'epochs', 'validation_loss',
                      'runtime', 'num_iterations']

    val_evaluation_params = []
    for M in params['M']:
        for K in params['K']:
            for learning_rate in params['learning_rate']:
                for epoch in params['epoch']:
                    for batch_size in params['batch_size']:
                        CV_log_sub = CV_log[(CV_log['margin'] == str(M)) & (CV_log['knowledge'] == str(K)) &
                                            (CV_log['learning_rate'] == str(learning_rate)) &
                                            (CV_log['epochs'] == str(epoch)) & (CV_log['batch_size'] == str(batch_size))]
                        print(CV_log_sub)
                        if CV_log_sub.shape[0] > 0:
                            val_evaluation_params.append([M, K, learning_rate, epoch, batch_size,
                                                          sum(list(CV_log_sub['validation_loss'])) /
                                                          CV_log_sub.shape[0]])
    selected_parameters = min(val_evaluation_params, key=lambda x: x[-1])

    val_evaluation_params = pd.DataFrame(val_evaluation_params)
    val_evaluation_params.columns = ['margin', 'knowledge', 'learning rate', 'epochs', 'batch_size',
                                     'average validation loss']
    # val_evaluation_params.to_csv('cross_validation_output.csv')
    print(val_evaluation_params)

    return all_models, val_evaluation_params, selected_parameters, CV_log
