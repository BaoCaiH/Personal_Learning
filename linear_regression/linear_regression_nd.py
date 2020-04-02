#!/usr/local/bin/python3
# coding: utf-8
"""Linear Regression nD @ Udemy."""

# %%
# Required libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# %%
def linear_formula(X, Y):
    """
    Return a coefficient and b as y-intercept.

    Input:
        X: numpy array
            x points
        Y: numpy array
            y points
    Output:
        a: float
            the coefficient of the line
        b: float
            y-intercept
    """
    denominator = X.dot(X) - X.mean()*X.sum()
    a = (X.dot(Y) - X.mean()*Y.sum())/denominator
    b = (X.dot(X)*Y.mean() - X.dot(Y)*X.mean())/denominator
    return (a, b)


def r_square(Y, Y_hat):
    """
    Return a coefficient and b as y-intercept.

    Input:
        Y: numpy array
            Y points
        Y_hat: numpy array
            y predicted points
    Output:
        r_s: float
            r square
    """
    diff_1 = Y - Y_hat
    diff_2 = Y - Y.mean()
    return 1 - diff_1.dot(diff_1)/diff_2.dot(diff_2)


# %%
X = []
Y = []
with open('data_2d.csv') as f:
    for line in f:
        x1, x2, y = line.split(',')
        X.append([float(x1), float(x2), 1])
        Y.append(float(y))
X = np.array(X)
Y = np.array(Y)
# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()
# %%
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Y_hat = np.dot(X, w)
print(r_square(Y, Y_hat))
# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)
ax.scatter(X[:, 0], X[:, 1], Y_hat)
plt.show()
# %%
X = []
Y = []
with open('data_poly.csv') as f:
    for line in f:
        x, y = line.split(',')
        X.append([1, float(x), float(x)**2])
        Y.append(float(y))
X = np.array(X)
Y = np.array(Y)
# %%
fig = plt.figure()
plt.scatter(X[:, 1], Y)
# plt.scatter(X[:, 2], Y)
plt.show()
# %%
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Y_hat = np.dot(X, w)
print(r_square(Y, Y_hat))
# %%
plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]), sorted(Y_hat))
plt.show()
