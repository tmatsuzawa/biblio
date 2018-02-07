import numpy as np



def curl(ux, uy):
    """
    Calculate curl of 2D field
    Parameters
    ----------
    var: 2d array
        element of var must be 2d array

    Returns
    -------

    """

    #ux, uy = var[0], var[1]
    xx, yy = ux.shape[0], uy.shape[1]

    omega = np.zeros((xx, yy))
    duxdx = np.gradient(ux, axis=0)
    duxdy = np.gradient(ux, axis=1)
    duydx = np.gradient(uy, axis=0)
    duydy = np.gradient(uy, axis=1)

    for i in range(xx):
        for j in range(yy):
            omega[i][j] = duxdy[i][j] - duydx[i][j]
    return omega

