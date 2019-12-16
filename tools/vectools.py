import numpy as np
import sys

def get_rate_of_strain_tensor(udata):
    """
    Assumes udata has a shape (d, nrows, ncols)
    ... one can easily make udata by np.stack((ux, uy))
    Parameters
    ----------
    udata: numpy array with shape (ux, uy) or (ux, uy, uz)
        ... assumes ux/uy/uz has a shape (nrows, ncols) or (nrows, ncols, nstacks)

    Returns
    -------
    sij: numpy array with shape (nrows, ncols, 2, 2) (dim=2) or (nrows, ncols, nstacks, 3, 3) (dim=3)
        ... idea is... sij[spacial coordinates, tensor indices]
            e.g.-  sij(x,y) = sij[y, x, i, j]

    """
    shape = udata.shape #shape=(dim, nrows, ncols, nstacks) if nstacks=0, shape=(dim, nrows, ncols)
    if shape[0] == 2:
        dim, nrows, ncols = udata.shape
        ux, uy = udata[0], udata[1]
        duxdx = np.gradient(ux, axis=1)
        duxdy = np.gradient(ux, axis=0)
        duydx = np.gradient(uy, axis=1)
        duydy = np.gradient(uy, axis=0)
        sij = np.zeros((nrows, ncols, dim, dim))
        sij[..., 0, 0] = duxdx
        sij[..., 0, 1] = duxdy
        sij[..., 1, 0] = duydx
        sij[..., 1, 1] = duydy
    elif shape[0] == 3:
        dim = 3
        ux, uy, uz = udata[0], udata[1], udata[2]
        try:
            nrows, ncols, nstacks = ux.shape
        except:
            print('ux.shape:')
            print(ux.shape)
            print('Space for which velocity is given must have a dimension at least (2,2,2) to take a gradient!!!')
            sys.exit(1)
        duxdx = np.gradient(ux, axis=1)
        duxdy = np.gradient(ux, axis=0)
        duxdz = np.gradient(ux, axis=2)
        duydx = np.gradient(uy, axis=1)
        duydy = np.gradient(uy, axis=0)
        duydz = np.gradient(uy, axis=2)
        duzdx = np.gradient(uz, axis=1)
        duzdy = np.gradient(uz, axis=0)
        duzdz = np.gradient(uz, axis=2)

        sij = np.zeros((nrows, ncols, nstacks, dim, dim))
        sij[..., 0, 0] = duxdx
        sij[..., 0, 1] = duxdy
        sij[..., 0, 2] = duxdz
        sij[..., 1, 0] = duydx
        sij[..., 1, 1] = duydy
        sij[..., 1, 2] = duydz
        sij[..., 2, 0] = duzdx
        sij[..., 2, 1] = duzdy
        sij[..., 2, 2] = duzdz
    elif shape[0] > 3:
        print('Not implemented yet.')
        return None
    return sij

def decompose_rate_of_strain_tensor(sij):
    """
    Decompose a rate of strain tensor into a symmetric and antisymmetric parts
    Parameters
    ----------
    sij

    Returns
    -------

    """
    dim = len(sij.shape) - 2 # spatial dim

    eij = np.zeros(sij.shape)
    # gij = np.zeros(sij.shape) #anti-symmetric part
    for i in range(dim):
        for j in range(dim):
            if j >= i:
                eij[..., i, j] += 1./2. * (sij[..., j, i] + sij[..., i, j])
                # gij[..., i, j] += 1./2. * (sij[..., j, i] - sij[..., i, j]) #anti-symmetric part
            else:
                eij[..., i, j] = eij[..., j, i]
                # gij[..., i, j] = -gij[..., j, i] #anti-symmetric part
    gij = sij - eij
    return eij, gij



def curl(udata):
    """
    Compute a curl of a velocity field using a rate of strain tensor
    ... For dim=3, the sign might need to be flipped... not tested
    ... if you already have velocity data as ux = array with shape (m, n) and uy = array with shape (m, n),
        udata = np.stack((ugrid1, vgrid1))
        omega = vec.curl(udata)
    Parameters
    ----------
    udata: (ux, uy)

    Returns
    -------

    """
    sij = get_rate_of_strain_tensor(udata)
    dim = len(sij.shape) - 2  # spatial dim
    eij, gij = decompose_rate_of_strain_tensor(sij)
    if dim == 2:
        omega = 2 * gij[..., 1, 0]
    elif dim == 3:
        # sign might be wrong
        omega1, omega2, omega3 = 2.* gij[..., 1, 2], 2.* gij[..., 2, 0], 2.* gij[..., 0, 1]
        omega = np.stack((omega1, omega2, omega3))
    else:
        print('Not implemented yet!')
        return None
    return omega





def curl_2d(ux, uy):
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
    # duxdx = np.gradient(ux, axis=1)
    duxdy = np.gradient(ux, axis=0)
    duydx = np.gradient(uy, axis=1)
    # duydy = np.gradient(uy, axis=0)

    omega = duydx - duxdy

    return omega

