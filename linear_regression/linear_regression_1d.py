#!/usr/bin/env python
# coding: utf-8
"""Linear Regression 1D @ Udemy."""

# %%
# Required libraries
import numpy as np
import matplotlib.pyplot as plt


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
# Read data in for scatter
X = []
Y = []
with open('data_1d.csv') as f:
    for line in f:
        x, y = line.split(',')
        X.append(float(x))
        Y.append(float(y))

# %%
X = np.array(X)
Y = np.array(Y)
a, b = linear_formula(X, Y)
Y_hat = a*X + b
print(r_square(Y, Y_hat))

# %%
plt.scatter(X, Y)
plt.plot(X, Y_hat)
plt.show()
