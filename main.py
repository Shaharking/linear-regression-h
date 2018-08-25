import pandas as pd
import numpy as np
import costFunction

def normalize_data(data):
    #return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    return (data - np.mean(data, axis=0)) / 255.

def load_data(path, lines_to_take = 5000):
    df = np.array(pd.read_csv(path, nrows=lines_to_take))
    labels = df[:, 0]
    labels = np.reshape(labels, (len(labels), 1))

    data = df[:, 1:]
    data = normalize_data(data)
    # We added bias to the data, we should not include column 0 when redisplay the images.
    data = np.c_[np.ones(labels.shape[0]), data]
    return labels, data

def sperate_data_to_train_and_cv(data, labels):
    # Separate to train and cross validation
    percentage_of_train = 0.7
    percentage_of_cv = 1 - percentage_of_train
    num_of_data_rows = labels.shape[0]
    num_of_train_rows = int(num_of_data_rows * percentage_of_train)
    train_data = data[0:num_of_train_rows, :]
    train_labels = labels[0:num_of_train_rows]
    cv_data = data[num_of_train_rows:, :]
    cv_labels = labels[num_of_train_rows:]
    return train_data, train_labels, cv_data, cv_labels

def main():
    train_path = './Data/mnist_train.csv'
    test_path = './Data/mnist_test.csv'

    labels, data = load_data(train_path)
    train_data, train_labels, cv_data, cv_labels = sperate_data_to_train_and_cv(data, labels)

    num_of_iterations = 400

    # we initial theta to be zeros.
    # theta = np.random.rand(train_data.shape[1], 1)
    theta = np.zeros((train_data.shape[1], 10))
    alpha = 3.6

    for digit in range(10):
        current_labels = np.array(1 * (train_labels == digit))
        temp_theta = np.zeros((train_data.shape[1], 1))
        for i in range(num_of_iterations):
            loss, gradients = costFunction.linear_regression_loss_function(temp_theta, train_data, current_labels, 0.1)
            temp_theta = temp_theta - (alpha*gradients)
            #print 'Iter %d with Cost %f for digit %d' % (i, loss, digit)

        # Calculate Errors on cross validation - useful for taking the best alpha and gama
        #    current_labels = np.array(1 * (cv_labels == digit))
        #    calculated_theta = temp_theta.reshape((len(temp_theta), 1))
        #    loss, gradients = costFunction.linear_regression_loss_function(calculated_theta, cv_data, current_labels, 0.)
        #    print 'Cost %f for digit %d on CROSS VALIDATION' %(loss, digit)

        print 'Cost %f for digit %d' % (loss, digit)
        theta[:, digit] = temp_theta.reshape((len(temp_theta)))

    results = np.dot(cv_data, theta)
    indexs = np.argmax(results, axis=1).reshape((len(cv_labels), 1))
    total_success = np.sum(indexs == cv_labels)
    percentage = (total_success / (1. * len(cv_labels)))
    print 'Total Successes %d from %d (Success Rate of - %f percentage)'%(total_success, len(cv_labels), percentage*100)



main()