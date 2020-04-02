#!/usr/bin/env python
# coding: utf-8
"""Moore's Law Demonstration in Python @ Udemy."""

# %%
# Required libraries
import re
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
# Read file
non_decimal = re.compile(r'[^\d]+')

X = []
Y = []
with open('moore.csv') as f:
    for line in f:
        txt = line.split('\t')
        x = int(non_decimal.sub('', txt[2].split('[')[0]))
        y = int(non_decimal.sub('', txt[1].split('[')[0]))
        X.append(x)
        Y.append(y)
# %%
X = np.array(X)
Y = np.array(Y)
# %%
Y_log = np.log(Y)
a, b = linear_formula(X, Y_log)
Y_log_hat = a*X + b

# <markdown>
# log(y1) = ax1 + b
# y1 = exp(ax1)*exp(b)
# 2y1 = 2exp(ax1)*2exp(b)
# 2y1 = exp(ln(2))*exp(ax1)*exp(b)
# exp(ax2)*exp(b) = exp(b)*exp(ax1 + ln(2))
# --> ax2 = ax1 + ln(2)
# a(x2 - x1) = ln(2)
# x2 - x1 = ln(2)/a
# x2 = x1 + ln(2)/a
# %%
print(r_square(Y_log, Y_log_hat))
print('It takes {} years for number of the transitors to double'.format(
    np.log(2)/a
))
plt.scatter(X, Y_log)
plt.plot(X, Y_log_hat)
plt.show()
