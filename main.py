import pandas as pd
import numpy as np
import costFunction
import itertools
import os.path
import matplotlib.pyplot as plt
import random
import cv2

def un_normalize_data(data, mean):
    return (255. * data) + mean

def normalize_data(data):
    #return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    # mean normalization
    mean = np.mean(data, axis=0)
    return (data - mean) / 255., mean

def load_data(path, lines_to_take = 5000):
    df = np.array(pd.read_csv(path, nrows=lines_to_take))
    labels = df[:, 0]
    labels = np.reshape(labels, (len(labels), 1))

    data = df[:, 1:]
    original_data = np.array(data)
    data, mean_normalize = normalize_data(data)
    # We added bias to the data, we should not include column 0 when redisplay the images.
    data = np.c_[np.ones(labels.shape[0]), data]
    return labels, data, original_data

def sperate_data_to_train_and_cv(data, labels, original_data):
    # Separate to train and cross validation
    percentage_of_train = 0.7
    percentage_of_cv = 1 - percentage_of_train
    num_of_data_rows = labels.shape[0]
    num_of_train_rows = int(num_of_data_rows * percentage_of_train)
    train_data = data[0:num_of_train_rows, :]
    train_labels = labels[0:num_of_train_rows]
    cv_data = data[num_of_train_rows:, :]
    cv_labels = labels[num_of_train_rows:]

    original_train_data = original_data[0:num_of_train_rows, :]
    original_cv_data = original_data[num_of_train_rows:, :]
    return train_data, train_labels, cv_data, cv_labels, original_train_data, original_cv_data

def main():
    train_path = './Data/mnist_train.csv'
    test_path = './Data/mnist_test.csv'
    matrix_path = 'matrix.txt'

    labels, data, original_data = load_data(train_path)
    train_data, train_labels, cv_data, cv_labels, original_train_data, original_cv_data = sperate_data_to_train_and_cv(data, labels, original_data)

    num_of_iterations = 400
    skip_training = False
    # we initial theta to be zeros.
    # theta = np.random.rand(train_data.shape[1], 1)
    theta = np.zeros((train_data.shape[1], 10))
    # alpha = 3.6

    if os.path.exists(matrix_path):
        theta = np.loadtxt(matrix_path)
        skip_training = True

    alhpas = [0.03, 0.09, 0.3, 0.9, 3.0, 6.0]
    lamdas = [0.03, 0.09, 0.3, 0.9, 3.0, 6.0]

    alhpas_lamdas = [i for i in itertools.product(alhpas, lamdas)]

    if not skip_training:

        for digit in range(10):
            current_labels = np.array(1 * (train_labels == digit))
            cvs_labels = np.array(1 * (cv_labels == digit))
            best_theta = np.zeros((train_data.shape[1], 1))
            min_loss = 100.0
            for (alpha, lamda) in alhpas_lamdas:
                temp_theta = np.zeros((train_data.shape[1], 1))
                for i in range(num_of_iterations):
                    loss, gradients = costFunction.linear_regression_loss_function(temp_theta, train_data, current_labels, lamda)
                    temp_theta = temp_theta - (alpha*gradients)

                    if loss > 1.2:
                        break
                    #print 'Iter %d with Cost %f for digit %d' % (i, loss, digit)

                # Calculate Errors on cross validation - useful for taking the best alpha and gama
                calculated_theta = temp_theta.reshape((len(temp_theta), 1))
                loss, gradients = costFunction.linear_regression_loss_function(calculated_theta, cv_data, cvs_labels, 0.)

                if loss < min_loss:
                    min_loss = loss
                    best_theta = temp_theta

                print 'Cost %f for digit %d on CROSS VALIDATION' %(loss, digit)

            print 'Cost %f for digit %d' % (loss, digit)
            theta[:, digit] = best_theta.reshape((len(best_theta)))

        np.savetxt('matrix.txt', theta)

    results = np.dot(cv_data, theta)
    indexs = np.argmax(results, axis=1).reshape((len(cv_labels), 1))
    total_success = np.sum(indexs == cv_labels)
    percentage = (total_success / (1. * len(cv_labels)))
    print 'Total Successes %d from %d (Success Rate of - %f percentage)'%(total_success, len(cv_labels), percentage*100)

    # Generate Figure of Pictures with Predicted Value
    random_images_indexs = random.sample(range(len(cv_labels)), 144)
    f, axarr = plt.subplots(12, 12)
    for idx in range(144):
        image_number = random_images_indexs[idx]
        is_success = np.sum(cv_labels[image_number] == indexs[image_number]) == 1
        digit_as_img = np.reshape(original_cv_data[image_number, :], (28, 28))
        color_map = plt.get_cmap('Greens')
        if not is_success:
            color_map = plt.get_cmap('Reds')
        axarr[idx / 12, idx % 12].imshow(digit_as_img, cmap=color_map)
        axarr[idx / 12, idx % 12].axis('off')

    plt.show(block=True)
    f.show()
    cv2.waitKey(500)


main()