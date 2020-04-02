#!/usr/local/bin/python3
# coding: utf-8
"""Linear Regression gradient descent @ Udemy."""

# %%
# Required libraries
import numpy as np
import matplotlib.pyplot as plt
# %%
N = 10
D = 3
# %%
X = np.zeros((N, D))
X[:, 0] = 1
X[:5, 1] = 1
X[5:, 2] = 1
# %%
X
# %%
Y = np.array([0]*5 + [1]*5)
# %%
Y
# %%
# Linearly solve for w
np.linalg.solve(X.T.dot(X), X.T.dot(Y))
# %%
costs = []
w = np.random.randn(D)/np.sqrt(D)
learning_rate = 0.001
# %%
for _ in range(1000):
    Y_hat = X.dot(w)
    delta = Y_hat - Y
    w = w - learning_rate*X.T.dot(delta)
    mse = delta.dot(delta)/N
    costs.append(mse)
# %%
plt.plot(costs)
plt.show()
# %%
print(w)
# %%
plt.plot(Y_hat, label='prediction')
plt.plot(Y, label='target')
plt.legend()
plt.show()
