import numpy as np
from scipy.optimize import curve_fit

###############################################################
# Standard functions
def linear_func(x, a, b):
    return a * x + b

def quad_func(x, a, b, c):
    return a*x**2 + b*x + c
def quad_func2(x, a, b, c):
    return a*(x-b)**2 + c

def power_func(x, a, b):
    return a * x ** b


def exp_func(x, a, b, c, x0):
    return a * np.exp(-b * (x-x0)) + c

def gaussian(x, a, x0, sigma, b):
    return a * np.exp(- (x - x0) ** 2. / (2. * sigma)) + b

def double_gaussian(x, a1, x1, sigma1, b1, a2, x2, sigma2, b2):
    return a1 * np.exp(- (x - x1) ** 2. / (2. * sigma1)) + b1 + a2 * np.exp(- (x - x2) ** 2. / (2. * sigma2)) + b2


###############################################################

def fit(func, xdata, ydata):
    if func is None:
        popt, pcov = curve_fit(linear_func, xdata, ydata)
    else:
        popt, pcov = curve_fit(func, xdata, ydata)
    return popt, pcov
