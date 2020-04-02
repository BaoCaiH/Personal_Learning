#!/usr/local/bin/python3
# coding: utf-8
"""Linear Regression systolic @ Udemy."""

# %%
# Required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# %%
df = pd.read_excel('mlr02.xls')
df['random'] = None
for i in range(len(df)):
    df.iloc[i, 3] = np.random.randint(100)
df['ones'] = 1
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
# X1 is blood pressure
# X2 is age in years
# X3 is weight in pounds
df.head()
# %%
X = np.array(df.iloc[:, 1:])
Y = np.array(df.iloc[:, 0])
# %%
plt.scatter(X[:, 1], Y)
plt.show()
# %%
print(
    'age only prediction accuracy: ',
    r_square(df['X1'], df[['X2', 'ones']].dot(linear_formula(
        df[['X2', 'ones']], df['X1'])
    ))
)
# %%
print(
    'weight only prediction accuracy: ',
    r_square(df['X1'], df[['X3', 'ones']].dot(linear_formula(
        df[['X3', 'ones']], df['X1'])
    ))
)
# %%
print(
    'both prediction accuracy: ',
    r_square(df['X1'], df[['X2', 'X3', 'ones']].dot(linear_formula(
        df[['X2', 'X3', 'ones']], df['X1'])
    ))
)
# %%
print(
    'both with random prediction accuracy: ',
    r_square(df['X1'], df[['X2', 'X3', 'random', 'ones']].dot(linear_formula(
        df[['X2', 'X3', 'random', 'ones']], df['X1'])
    ))
)
