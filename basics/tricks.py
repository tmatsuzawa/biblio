"""
Module for nice tricks that are easy to forget in coding
"""
import numpy as np


def roundup2pow10(x):
    """
    Rounds up to the closest power of 10 such that x < 10**n
    e.g.- 424 -> 1000, 1.2457 -> 10
    Parameters
    ----------
    x

    Returns
    -------

    """
    return 10**np.ceil(np.log10(x))


def roundup(x):
    exponent = np.floor(np.log10(x))
    number = np.ceil(10 ** (np.log10(x) - exponent))
    return number * 10 ** exponent


