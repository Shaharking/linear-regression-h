import numpy as np
from sigmoid import sigmoid

def linear_regression_loss_function(theta, X, y, lamda):

    m = len(y)

    loss = 0
    gradients = np.zeros(theta.shape)

    g = np.dot(X, theta)
    h = sigmoid(g)
    # loss = np.mean(np.multiply(-y , np.log(h)) - np.multiply((1 - y), np.log(1-h)))
    loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    loss_regularization_term = ( lamda * np.sum(np.power(theta[1:], 2)) ) / (2 * m)
    loss = loss + loss_regularization_term

    gradients = np.sum(np.dot(np.transpose(X), (h - y)), axis=0) / m
    copy_theta = theta
    copy_theta[0] = 0
    loss_gradients_term = (lamda * copy_theta) / m
    gradients = gradients + loss_gradients_term

    return loss, gradients





