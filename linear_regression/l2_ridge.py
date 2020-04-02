#!/usr/local/bin/python3
# coding: utf-8
"""Linear Regression ridge @ Udemy."""

# %%
# Required libraries
import numpy as np
import matplotlib.pyplot as plt
# %%
X = np.linspace(0, 10, 100)
Y = 0.5*X + np.random.randn(100)
Y[-1] += 20
Y[-2] += 20
plt.scatter(X, Y)
plt.show()
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
        w: float
            the coefficient of the line
    """
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))


def l2_ridge(X, Y, l2):
    """Return linear but with ridge."""
    punished = X.T.dot(X) + np.eye(len(X[0]))*l2
    return np.linalg.solve(punished, np.dot(X.T, Y))


# %%
X = np.array([np.ones(100), X]).T
# %%
w = linear_formula(X, Y)
Y_hat = X.dot(w)
l2 = 1000
w_l2 = l2_ridge(X, Y, l2)
Y_hat_ridge = X.dot(w_l2)
plt.scatter(X.T[1], Y)
plt.plot(X.T[1], Y_hat, label='Normal')
plt.plot(X.T[1], Y_hat_ridge, label='Ridge')
plt.legend()
plt.show()
