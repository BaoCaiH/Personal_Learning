#!/usr/local/bin/python3
# coding: utf-8
"""Linear Regression Overfitting @ Udemy."""

# %%
# Required libraries
import numpy as np
import matplotlib.pyplot as plt
# %%
N = 100
X = np.linspace(0, 8*np.pi, N)
Y = np.sin(X)
# %%
plt.plot(X, Y)
plt.show()


# %%
def poly_data(X, deg):
    """Create poly degrees data."""
    n = len(X)
    data = [np.ones(n)]
    for i in range(1, deg+1):
        data.append(X**i)
    return np.array(data).T


def fit(X, Y):
    """Fit a linear regression."""
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))


def fit_display(X, Y, sample, deg):
    """Fit and display on graph."""
    N = len(X)
    train_id = np.random.choice(N, sample)
    X_train = X[train_id]
    Y_train = Y[train_id]

    plt.scatter(X_train, Y_train)
    plt.show()

    X_train_poly = poly_data(X_train, deg)
    w = fit(X_train_poly, Y_train)

    X_poly = poly_data(X, deg)
    Y_hat = X_poly.dot(w)
    plt.plot(X, Y)
    plt.plot(X, Y_hat)
    plt.scatter(X_train, Y_train)
    plt.title('deg: {}'.format(deg))
    plt.show()


def mean_squared_error(Y, Y_hat):
    """Return mean squared error."""
    error = Y - Y_hat
    return error.dot(error)/len(error)


def train_and_test(X, Y, sample=20, max_deg=20):
    """Plot train and test curve together."""
    N = len(X)
    train_id = np.random.choice(N, sample)
    X_train = X[train_id]
    Y_train = Y[train_id]
    test_id = [i for i in range(N) if i not in train_id]
    X_test = X[test_id]
    Y_test = Y[test_id]

    mse_train = []
    mse_test = []
    for i in range(1, max_deg+1):
        X_poly_train = poly_data(X_train, i)
        X_poly_test = poly_data(X_test, i)
        w = fit(X_poly_train, Y_train)
        Y_hat_train = X_poly_train.dot(w)
        Y_hat_test = X_poly_test.dot(w)
        mse_train.append(mean_squared_error(Y_train, Y_hat_train))
        mse_test.append(mean_squared_error(Y_test, Y_hat_test))
    plt.plot(np.linspace(1, max_deg, max_deg), mse_train)
    plt.plot(np.linspace(1, max_deg, max_deg), mse_test)
    plt.title('train and test')
    plt.show()
    plt.plot(np.linspace(1, max_deg, max_deg), mse_train)
    plt.title('only train')
    plt.show()


# %%
train_and_test(X, Y)
