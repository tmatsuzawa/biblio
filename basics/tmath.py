import numpy as np
# Module for elementary mathematical operations


### SEQUENCE ###
def recursive_eqaution(func, x0, n=100, **kwargs):
    """
    Returns the elements defined by a recursive equation

    e.g.- logistic equation
    n = 50
    f = lambda x, r: r * x * (1 - x)
    xn = recursive_eqaution(f, x0=0.1, r=2.5, n=n)

    Parameters
    ----------
    func: function
        ... recursive equation
    x0: initial value, default
    n: int, number of iterations
    kwargs: keyword arguments
        ... the unknown keyword arguments will be passed to func

    Returns
    -------
    xn: array which stores  values like [x0, x1, x2, ..., xn]

    """
    xn = []
    xn.append(x0)
    for i in range(n):
        xn.append(func(xn[-1], **kwargs))
    return np.asarray(xn)



def get_converged_pt(x, y, tol=0.1, verbose=True):
    """
    DEPRECIATED: Use get_fixed_pts for the same but more general usage. This function is a backbone of get_fixed_pts.


    Given that x and y are sequences and converge into a point (xf, yf) as n goes infinity,
    this function returns xf, yf.

    The numerical convergence is determined by whether (x[i]-x[i-i], y[i]-y[i-1) falls inside a radius "tol"

    If the sequences do not converge, it returns (np.nan, np.nan)

    Parameters
    ----------
    x: array
        ... sequence 1
    y: array
        ... sequence 1
    tol: float
        ... radius to determine the convergence. the smaller the value, the stricter the convergence is defined.
    verbose: bool
        ... If true, it outputs, the number of iterations it took to converge

    Returns
    -------

    """
    l = len(x)
    for i in range(1, l):
        x_cand, y_cand = x[i - 1], y[i - 1]
        deltaX, deltaY = x[i] - x_cand, y[i] - y_cand
        deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2)

        if deltaR < tol:
            if verbose:
                print('... No of iterrations it took to converge, tolerance: %d, %f' % (i, tol))
            x_cnv, y_cnv = x[i], y[i]
            break

        if i == l - 1:
            print('... the sequence did not converge after %d iterations. Consider changing the tolerance %f' % (
            i, tol))
            x_cnv, y_cnv = np.nan, np.nan
    return x_cnv, y_cnv


def get_fixed_pts(x, y, tol=0.01, verbose=False):
    """
    Given that x and y are sequences and converge into a point (xf, yf) as n goes infinity
    OR x and y are PERIODIC sequences with fixed points (xp, yp),
    this function returns (xf, yf) and (xp, yp).

    This is a natural extension of get_converged_pt().

    The numerical convergence is determined by whether (x[i]-x[i-i], y[i]-y[i-1]) falls inside a radius "tol"

    If the sequences do not converge, it returns (np.nan, np.nan)

    Parameters
    ----------
    x: array
        ... sequence 1
    y: array
        ... sequence 1
    tol: float
        ... radius to determine the convergence. the smaller the value, the stricter the convergence is defined.
    verbose: bool
        ... If true, it outputs, the number of iterations it took to converge

    Returns
    -------

    """

    l = len(x)
    periods = list(range(1, int(l / 2)))

    for i, period in enumerate(periods):
        xfs, yfs = np.empty(period), np.empty(period)
        xfs[:] = np.nan
        yfs[:] = np.nan
        for j in range(period):
            x_tmp, y_tmp = x[j:][::period], y[j:][::period]
            x_cnv, y_cnv = get_converged_pt(x_tmp, y_tmp, tol=tol, verbose=verbose)
            if not np.isnan(x_cnv) and not np.isnan(y_cnv):
                xfs[j], yfs[j] = x_cnv, y_cnv
                if j == period - 1:
                    return xfs, yfs
            else:
                if period == periods[-1]:
                    print('... Did not converge! Returning None, None...')
                    return None, None
                else:
                    # Move onto the next period
                    break


def get_dst_from_pt_to_line(x0, y0, a, b, c, signed=False):
    """
    Returns a distance between a pt and a line

    Eq of a line: ax + by + c = 0
    Point is at (x0, y0)
    Parameters
    ----------
    x
    y
    a
    b
    c

    Returns
    -------

    """
    if not signed:
        d = np.abs( ( a * x0 + b * y0 + c ) ) / np.sqrt(a ** 2 + b ** 2)
    else:
        d = (a * x0 + b * y0 + c) / np.sqrt(a ** 2 + b ** 2)
    return d

def get_pt_projected_onto_line(x0, y0, a, b, c):
    """
    Assume a line and a point, and project the point onto the line.
    ... In other words, draw a circle with a radius (x0, y0) with d which is a distance between the pt and the line.
        Then, this function returns the intersection of the circle with the line.
    ... Eq of a line: ax + by + c = 0
        Point is at (x0, y0)

    Parameters
    ----------
    x0
    y0
    a
    b
    c

    Returns
    -------

    """
    d = get_dst_from_pt_to_line(x0, y0, a, b, c, signed=True)
    # theta = np.arctan2(-a, b)
    # x1 = x0 - d * np.cos(theta)
    # y1 = y0 + d * np.sin(theta)

    x1 = x0 - a * d / np.sqrt(a**2 + b**2)
    y1 = y0 - b * d / np.sqrt(a**2 + b**2)

    return x1, y1