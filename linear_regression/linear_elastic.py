#!/usr/local/bin/python3
# coding: utf-8
"""Linear Regression elastic net @ Udemy."""

# %%
# Required libraries
import numpy as np
import matplotlib.pyplot as plt
# %%
N = 50
D = 50
X = (np.random.random((N, D)) - 0.5)*10
Y = X.dot(np.array([-0.3, 0.9, 0.5, -0.1]+[0]*(D-4))) + np.random.randn(N)+0.5
# %%


class LinearRegression:
    """Linear regression model."""

    def __init__(self, l1_r, lmbd, learning_rate, max_iter=10000):
        """Initialize a model."""
        self.l1_r = l1_r
        self.lmbd = lmbd
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit(self, X, Y):
        """Fit a linear model."""
        costs = []
        w = np.random.randn(len(X[0]))/np.sqrt(len(X[0]))
        for _ in range(self.max_iter):
            Y_hat = X.dot(w)
            delta = Y_hat - Y
            w = w - self.learning_rate*(
                X.T.dot(delta)
                + self.l1_r*self.lmbd*np.sign(w)
                + 2*(1 - self.l1_r)*self.lmbd*w
            )
            mse = delta.dot(delta)/len(X)
            costs.append(mse)
        plt.plot(costs)
        plt.show()
        self.w = w

    def predict(self, X):
        """Return Y hat."""
        return X.dot(self.w)

    def score(self, X, Y):
        """Return r square score."""
        Y_hat = X.dot(self.w)
        delta = Y - Y_hat
        delta_mean = Y - Y.mean()
        return 1 - delta.dot(delta)/delta_mean.dot(delta_mean)


# %%
linear = LinearRegression(0.4, 10, 0.0001, 1000)
linear.fit(X, Y)
linear.score(X, Y)
