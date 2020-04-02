#!/usr/local/bin/python3
# coding: utf-8
"""Linear Regression lasso @ Udemy."""

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


def l1_lasso(X, Y, l1, learning_rate, max_iter=10000):
    """Return linear but with lasso."""
    costs = []
    w = np.random.randn(len(X[0]))/np.sqrt(len(X[0]))
    for _ in range(max_iter):
        Y_hat = X.dot(w)
        delta = Y_hat - Y
        w = w - learning_rate*(X.T.dot(delta) + l1*np.sign(w))
        mse = delta.dot(delta)/len(X)
        costs.append(mse)
    print(w)
    plt.plot(costs)
    plt.show()
    return w


# %%
w = l1_lasso(X, Y, 10, 0.001, 100)
# %%
plt.plot(np.array([-0.3, 0.9, 0.5, -0.1]+[0]*(D-4)), label='true')
plt.plot(w, label='calculated')
plt.legend()
plt.show()
