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



# Vortex ring: formation number (Gharib)
def compute_form_no(span, orifice_d=25.6, piston_d=160., num_orifices=1):
    LD = (piston_d / orifice_d)**2 * span / orifice_d / num_orifices
    return LD