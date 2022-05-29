from Supplements import *
from NeuralNetsLoanEligibility import *
from matplotlib.offsetbox import AnchoredText


if __name__ == "__main__":
    start = time.time()
    dataset = 'loan_data_set.csv'
    data = pd.read_csv(dataset, header=0)
    processed_output = data_processing(data)
    data = processed_output[0]
    X = data.iloc[:, 0:data.shape[1] - 1]
    y_original = list(data.iloc[:, -1])
    scaled_income = processed_output[1].scale_[0] * (5000 - processed_output[1].data_min_[0])
    X, y = data_manipulation(X, y_original)
    K = [scaled_income, 0]
    K = None
    sizes = [X.shape[1], 4, 4, 1]
    batch_size = 10
    epochs = 5
    learning_rate = 0.1
    model = NeuralNetLoanEligibility(sizes=sizes)
    model.learn(np.array(X), y, K=K, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)
    print(model.training_accuracies)
    print(model.training_losses)
    sys.exit(0)
    plt.figure()
    plt.subplot(211)
    plt.plot(model.training_accuracies)
    plt.annotate('layers: {}'.format(sizes), xy=(0.05, 1.05), xycoords='axes fraction')
    plt.ylim(top=1)
    plt.ylim(bottom=0)
    plt.xlabel('batches')
    plt.ylabel('training accuracy')
    plt.subplot(212)
    plt.plot(model.training_losses)
    plt.xlabel('batches')
    plt.ylabel('training entropy loss')

    filename = 'training_set_weight_constraint.png'
    plt.savefig(filename)
    plt.close()
