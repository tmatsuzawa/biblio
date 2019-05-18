from scipy.stats import binned_statistic
from tqdm import tqdm
import numpy as np
import sys
import library.basics.formatarray as fa
import library.tools.rw_data as rw
import os
import numpy.ma as ma

"""
Philosophy:
udata = (ux, uy, uz) or (ux, uy)
each ui has a shape (height, width, (depth), duration)

If ui's are individually given, make udata like 
udata = np.stack((ux, uy))
"""


# Fundamental operations
def get_rate_of_strain_tensor(udata):
    """
    Assumes udata has a shape (d, nrows, ncols, duration) or  (d, nrows, ncols)
    ... one can easily make udata by np.stack((ux, uy))
    Parameters
    ----------
    udata: numpy array with shape (ux, uy) or (ux, uy, uz)
        ... assumes ux/uy/uz has a shape (nrows, ncols, duration) or (nrows, ncols, nstacks, duration)
        ... can handle udata without temporal axis

    Returns
    -------
    sij: numpy array with shape (nrows, ncols, 2, 2) (dim=2) or (nrows, ncols, nstacks, duration, 3, 3) (dim=3)
        ... idea is... sij[spacial coordinates, time, tensor indices]
            e.g.-  sij(x, y, t) = sij[y, x, t, i, j]
        ... sij = d ui / dxj


    """
    shape = udata.shape #shape=(dim, nrows, ncols, nstacks) if nstacks=0, shape=(dim, nrows, ncols)
    if shape[0] == 2:
        ux, uy = udata[0, ...], udata[1, ...]
        try:
            dim, nrows, ncols, duration = udata.shape
        except:
            dim, nrows, ncols = udata.shape
            duration = 1
            ux = ux.reshape((ux.shape[0], ux.shape[1], duration))
            uy = uy.reshape((uy.shape[0], uy.shape[1], duration))

        duxdx = np.gradient(ux, axis=1)
        duxdy = np.gradient(ux, axis=0)
        duydx = np.gradient(uy, axis=1)
        duydy = np.gradient(uy, axis=0)
        sij = np.zeros((nrows, ncols, duration, dim, dim))
        sij[..., 0, 0] = duxdx
        sij[..., 0, 1] = duxdy
        sij[..., 1, 0] = duydx
        sij[..., 1, 1] = duydy
    elif shape[0] == 3:
        dim = 3
        ux, uy, uz = udata[0, ...], udata[1, ...], udata[2, ...]
        try:
            # print ux.shape
            nrows, ncols, nstacks, duration = ux.shape
        except:
            nrows, ncols, nstacks = ux.shape
            duration = 1
            ux = ux.reshape((ux.shape[0], ux.shape[1], ux.shape[2], duration))
            uy = uy.reshape((uy.shape[0], uy.shape[1], uy.shape[2], duration))
            uz = uz.reshape((uz.shape[0], uz.shape[1], uz.shape[2], duration))
        duxdx = np.gradient(ux, axis=1)
        duxdy = np.gradient(ux, axis=0)
        duxdz = np.gradient(ux, axis=2)
        duydx = np.gradient(uy, axis=1)
        duydy = np.gradient(uy, axis=0)
        duydz = np.gradient(uy, axis=2)
        duzdx = np.gradient(uz, axis=1)
        duzdy = np.gradient(uz, axis=0)
        duzdz = np.gradient(uz, axis=2)

        sij = np.zeros((nrows, ncols, nstacks, duration, dim, dim))
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
        print 'Not implemented yet.'
        return None
    return sij

def decompose_rate_of_strain_tensor(sij):
    """
    Decompose a rate of strain tensor into a symmetric and antisymmetric parts
    Parameters
    ----------
    sij, 5d or 6d numpy array (x, y, t, i, j) or (x, y, z, t, i, j)

    Returns
    -------
    eij: 5d or 6d numpy array, symmetric part of rate-of-strain tensor.
         5d if spatial dimensions are x and y. 6d if spatial dimensions are x, y, and z.
    gij: 5d or 6d numpy array, anti-symmetric part of rate-of-strain tensor.
         5d if spatial dimensions are x and y. 6d if spatial dimensions are x, y, and z.

    """
    dim = len(sij.shape) - 3 # spatial dim
    if dim == 2:
        duration = sij.shape[2]
    elif dim == 3:
        duration = sij.shape[3]

    eij = np.zeros(sij.shape)
    # gij = np.zeros(sij.shape) #anti-symmetric part
    for t in range(duration):
        for i in range(dim):
            for j in range(dim):
                if j >= i:
                    eij[..., t, i, j] = 1./2. * (sij[..., t, i, j] + sij[..., t, j, i])
                    # gij[..., i, j] += 1./2. * (sij[..., i, j] - sij[..., j, i]) #anti-symmetric part
                else:
                    eij[..., t, i, j] = eij[..., t, j, i]
                    # gij[..., i, j] = -gij[..., j, i] #anti-symmetric part

    gij = sij - eij
    return eij, gij

def reynolds_decomposition(udata):
    """
    Apply the Reynolds decomposition to a velocity field

    Parameters
    ----------
    udata: numpy array
          ... (ux, uy, uz) or (ux, uy)
          ... ui has a shape (height, width, depth, duration) or (height, width, depth) (3D)
          ... ui may have a shape (height, width, duration) or (height, width) (2D)

    Returns
    -------
    u_mean: nd array, mean velocity field
    u_turb: nd array, turbulent velocity field
    """
    udata = fix_udata_shape(udata)
    dim = len(udata)

    # Initialization
    u_mean = np.zeros_like(udata)
    u_turb = np.zeros_like(udata)
    for i in range(dim):
        u_mean[i] = np.nanmean(udata[i], axis=dim) # axis=dim is always the time axis in this convention
        u_turb[i] = udata[i] - u_mean[i]
    return u_mean, u_turb

# vector operations
def div(udata):
    """
    Computes divergence of a velocity field
    Parameters
    ----------
    udata: numpy array
          ... (ux, uy, uz) or (ux, uy)
          ... ui has a shape (height, width, depth, duration) or (height, width, depth) (3D)
          ... ui may have a shape (height, width, duration) or (height, width) (2D)

    Returns
    -------
    div_u: numpy array
          ... div_u has a shape (height, width, depth, duration) (3D) or (height, width, duration) (2D)
    """
    sij = get_rate_of_strain_tensor(udata) #shape (nrows, ncols, duration, 2, 2) (dim=2) or (nrows, ncols, nstacks, duration, 3, 3) (dim=3)
    dim = len(sij.shape) - 3  # spatial dim
    div_u = np.zeros(sij.shape[:-2])
    for d in range(dim):
        div_u += sij[..., d, d]
    return div_u

def curl(udata):
    """
    Computes curl of a velocity field using a rate of strain tensor
    ... For dim=3, the sign might need to be flipped... not tested
    ... if you already have velocity data as ux = array with shape (m, n) and uy = array with shape (m, n),
        udata = np.stack((ugrid1, vgrid1))
        omega = vec.curl(udata)
    Parameters
    ----------
    udata: (ux, uy, uz) or (ux, uy)

    Returns
    -------
    omega: numpy array with shape (height, width, duration) (2D) or (height, width, duration) (2D)

    """
    sij = get_rate_of_strain_tensor(udata)
    dim = len(sij.shape) - 3  # spatial dim
    eij, gij = decompose_rate_of_strain_tensor(sij)
    if dim == 2:
        omega = 2 * gij[..., 1, 0]
    elif dim == 3:
        # sign might be wrong
        omega1, omega2, omega3 = 2.* gij[..., 2, 1], 2.* gij[..., 0, 2], 2.* gij[..., 1, 0]
        omega = np.stack((omega1, omega2, omega3))
    else:
        print 'Not implemented yet!'
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

# Elementary analysis
def get_energy(udata):
    shape = udata.shape  # shape=(dim, nrows, ncols, nstacks) if nstacks=0, shape=(dim, nrows, ncols)
    dim = udata.shape[0]
    energy = np.zeros(shape[1:])
    for d in range(dim):
        energy += udata[d, ...] ** 2
    energy /= 2.
    return energy

def get_enstrophy(udata):
    dim = udata.shape[0]

    omega = curl(udata)
    shape = omega.shape # shape=(dim, nrows, ncols, nstacks, duration) if nstacks=0, shape=(dim, nrows, ncols, duration)
    enstrophy = np.zeros(shape[1:])

    for d in range(dim):
        enstrophy += omega[d, ...] ** 2
    enstrophy /= 2.
    return enstrophy

def fft_nd(field, dx=1, dy=1, dz=1):
    """
    Parameters
    ----------
    field: np array, (height, width, depth, duration) or (height, width, depth, duration)
    dx
    dy
    dz

    Returns
    -------

    """
    dim = len(field.shape) - 1

    field_fft = np.abs(np.fft.fftn(field, axes=range(dim)))
    field_fft = np.fft.fftshift(field_fft, axes=range(dim))

    if dim == 2:
        height, width, duration = field.shape
        lx, ly = dx * width, dy * height
        # volume = lx * ly

        ky, kx = np.mgrid[-height / 2: height / 2: complex(0, height),
                 -width / 2: width / 2: complex(0, width)]
        kx = kx * 2 * np.pi / lx
        ky = ky * 2 * np.pi / ly
        return field_fft, np.asarray([kx, ky])

    elif dim == 3:
        height, width, depth, duration = field.shape
        lx, ly, lz = dx * width, dy * height, dz * depth
        # volume = lx * ly * lz
        ky, kx, kz = np.mgrid[-height / 2: height / 2: complex(0, height),
                     -width / 2: width / 2: complex(0, width),
                     -depth / 2: depth / 2: complex(0, depth)]
        kx = kx * 2 * np.pi / (dx * width)
        ky = ky * 2 * np.pi / (dy * height)
        kz = kz * 2 * np.pi / (dz * height)
        return field_fft, np.asarray([kx, ky, kz])

def get_energy_spectrum_nd(udata, dx=1, dy=1, dz=1):
    """
    Returns nd energy spectrum from velocity data
    Parameters
    ----------
    udata
    dx: data spacing in x (units: mm/px)
    dy: data spacing in y (units: mm/px)
    dz: data spacing in z (units: mm/px)

    Returns
    -------

    """
    udata = fix_udata_shape(udata)

    energy = get_energy(udata)
    dim = len(udata)

    energy_fft = np.abs(np.fft.fftn(energy, axes=range(dim)))
    energy_fft = np.fft.fftshift(energy_fft, axes=range(dim))


    if dim == 2:
        height, width, duration = energy.shape
        lx, ly = dx * width, dy * height
        # volume = lx * ly

        kyy, kxx = np.mgrid[-height / 2: height / 2: complex(0, height),  #[-512, -511, ... , 511, 512]
                 -width / 2: width / 2: complex(0, width)]

        kx = 2. * np.pi * kxx / (2*np.pi * width)
        ky = 2. * np.pi * kyy / (2*np.pi * height)
        return energy_fft, np.asarray([kx, ky])

    elif dim == 3:
        height, width, depth, duration = energy.shape
        lx, ly, lz = dx * width, dy * height, dz * depth
        # volume = lx * ly * lz
        kyy, kxx, kzz = np.mgrid[-height / 2: height / 2: complex(0, height),
                     -width / 2: width / 2: complex(0, width),
                     -depth / 2: depth / 2: complex(0, depth)]
        kx = 2. * np.pi * kxx / (2*np.pi * width)
        ky = 2. * np.pi * kyy / (2*np.pi * height)
        kz = 2. * np.pi * kzz / (2*np.pi * depth)
        return energy_fft, np.asarray([kx, ky, kz])

def get_energy_spectrum(udata, z=0, dx=1, dy=1, dz=1, nkout=40):
    def delete_masked_elements(data, mask):
        """
        Deletes elements of data using mask, and returns a 1d array
        Parameters
        ----------
        data: N-d array
        mask: N-d array, bool

        Returns
        -------
        compressed_data

        """
        data_masked = ma.array(data, mask=mask)
        compressed_data = data_masked.compressed()
        '...Reduced data using a given mask'
        return compressed_data

    def convert_2d_spec_to_1d(s_e, kx, ky, nkout=40):
        """Convert a 2d spectrum s_e computed over a grid of k values kx and ky into a 1d spectrum computed over k

        Parameters
        ----------
        s_e : n x m float array
            The fourier transform intensity pattern to convert to 1d
        kx : n x m float array
            input wavenumbers' x components
        ky : n x m float array
            input wavenumbers' y components, as np.array([[y0, y0, y0, y0, ...], [y1, y1, y1, y1, ...], ...]])
        nkout : int
            approximate length of the k vector for output; will be larger since k values smaller than 1/L will be removed
            number of bins for the output k vector

        Returns
        -------
        s_k :
        kbin :
        """
        nx, ny,  nt = np.shape(s_e)

        kk = np.sqrt(np.reshape(kx ** 2 + ky ** 2, (nx * ny)))
        kx_1d = np.sqrt(np.reshape(kx ** 2, (nx * ny)))
        ky_1d = np.sqrt(np.reshape(ky ** 2, (nx * ny)))

        s_e = np.reshape(s_e, (nx * ny, nt))
        # sort k by values
        #    indices=np.argsort(k)
        #    s_e=s_e[indices]

        nk, nbin = np.histogram(kk, bins=nkout)
        #     print 'Fourier.spectrum_2d_to_1d_convert(): nkout = ', nkout
        #     print 'Fourier.spectrum_2d_to_1d_convert(): nbin = ', nbin
        nn = len(nbin) - 1

        # remove too small values of kx or ky (components aligned with the mesh directions)
        epsilon = nbin[0]
        kbin = np.zeros(nn)
        indices = np.zeros((nx * ny, nn), dtype=bool)
        jj = 0
        okinds_nn = []
        for ii in range(nn):
            #         print 'ii = ', ii
            indices[:, ii] = np.logical_and(np.logical_and(kk >= nbin[ii], kk < nbin[ii + 1]),
                                            np.logical_and(np.abs(kx_1d) >= epsilon, np.abs(ky_1d) >= epsilon))

            # If there are any values to add, add them to kbin (prevents adding values less than epsilon
            if indices[:, ii].any():
                kbin[jj] = np.mean(kk[indices[:, ii]])
                okinds_nn.append(ii)
                jj += 1

        s_k = np.zeros((nn, nt))
        # s_part = np.zeros(nx * ny)

        #     print('Fourier.spectrum_2d_to_1d_convert(): Compute 1d fft from 2d')
        for t in range(nt):
            s_part = s_e[:, t]
            jj = 0
            for ii in okinds_nn:
                s_k[jj, t] = np.nanmean(s_part[indices[:, ii]])
                jj += 1

        kbin = ma.masked_equal(kbin, 0)
        s_k = ma.masked_equal(s_k, 0)
        # kbin = kbin.compressed()

        return s_k, kbin


    e_ks, ks = get_energy_spectrum_nd(udata, dx=dx, dy=dy, dz=dz)

    if len(udata) == 3:
        kx, ky, kz = ks[0], ks[1], ks[2]
        ######################################################################
        # Currently, I haven't cleaned up a code to 3d power spectra into 1d.
        # The currently implemented solution is just use a 2D slice of the 3D data.
        ######################################################################
        kx, ky = kx[..., 0], ky[..., 0]
    elif len(udata) == 2:
        kx, ky = ks[0], ks[1]

    e_k, kk = convert_2d_spec_to_1d(e_ks[:, :, 0, :], kx, ky, nkout=nkout)



    return e_k, kk

def get_rescaled_energy_spectrum(udata, epsilon=10**5, nu=1.0034, z=0, dx=1, dy=1, dz=1, nkout=40):
    # get energy spectrum
    e_k, kk = get_energy_spectrum(udata, z=z, dx=dx, dy=dy, dz=dz, nkout=nkout)

    # Kolmogorov length scale
    eta = (nu ** 3 / epsilon) ** (0.25)  # mm

    k_norm =kk * eta
    e_k_norm = e_k[...] / ((epsilon * nu ** 5.) ** (0.25))
    return e_k_norm, k_norm

# advanced analysis
def compute_spatial_autocorr(ui, x, y, roll_axis=1, n_bins=None, x0=None, x1=None, y0=None, y1=None, t0=None, t1=None,
                             coarse=1.0):
    """
    Compute spatial autocorrelation function of 2+1 velocity field
    Spatial autocorrelation function:
        f = <u_j(\vec{x}) u_j(\vec{x}+r\hat{x_i})> / <u_j(\vec{x})^2>
    where velocity vector u = (u1, u2).
    If i = j, f is called longitudinal autocorrelation function.
    Otherwise, f is called transverse autocorrelation function.

    Example:
        u = ux # shape(height, width, duration)
        x_, y_  = range(u.shape[1]), range(u.shape[0])
        x, y = np.meshgrid(x_, y_)

        # LONGITUDINAL AUTOCORR FUNC
        rrs, corrs, corr_errs = compute_spatial_autocorr(u, x, y, roll_axis=1)  # roll_axis is i in the description above

        # TRANSVERSE AUTOCORR FUNC
        rrs, corrs, corr_errs = compute_spatial_autocorr(u, x, y, roll_axis=0)  # roll_axis is i in the description above


    Parameters
    ----------
    ui: numpy array, 2 + 1 scalar field. i.e. shape: (height, width, duration)
    x: numpy array, 2d grid
    y: numpy array, 2d grid
    roll_axis: int, axis index to compute correlation... 0 for y-axis, 1 for x-axis
    n_bins: int, number of bins to compute statistics
    x0: int, index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    x1: int, index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y0: int, index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y1: int, index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t0: int, index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t1: int, index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    coarse:
    Returns
    -------
    rr: 2d numpy array, (distance, time)
    corr: 2d numpy array, (autocorrelation values, time)
    corr_err: 2d numpy array, (std of autocorrelation values, time)
    """

    # Array sorting
    def sort2arr(arr1, arr2):
        """
        Sort arr1 and arr2 using the order of arr1
        e.g. a=[2,1,3], b=[9,1,4] -> a[1,2,3], b=[1,9,4]
        Parameters
        ----------
        arr1
        arr2

        Returns
        -------
        Sorted arr1, and arr2

        """
        arr1, arr2 = zip(*sorted(zip(arr1, arr2)))
        return arr1, arr2

    if x0 is None:  # if None, use the whole space
        x0, y0 = 0, 0
        x1, y1 = ui.shape[1], ui.shape[0]
    if t0 is None:
        t0 = 0
    if t1 is None:
        t1 = ui.shape[2]

    # Some useful numbers for processing
    nrows, ncolumns = y1 - y0, x1 - x0
    limits = [ncolumns, nrows]
    # Number of bins- if this is too small, correlation length would be overestimated. Keep it around ncolumns
    if n_bins is None:
        n_bins = ncolumns

    # Use a portion of data
    y_grid, x_grid = y[y0:y1, x0:x1], x[y0:y1, x0:x1]

    # Initialization
    rrs, corrs, corr_errs = np.empty((n_bins, t1 - t0)), np.empty((n_bins, t1 - t0)), np.empty((n_bins, t1 - t0))

    for t in tqdm(range(t0, t1), desc='time'):
        # Call velocity field at time t as uu
        uu = ui[y0:y1, x0:x1, t]

        uu2_norm = np.nanmean(ui[y0:y1, x0:x1, ...] ** 2, axis=(0, 1))  # mean square velocity

        # Initialization
        rr = np.empty((x_grid.size, int(coarse * limits[roll_axis]) * 2 - 1))
        corr = np.empty((x_grid.size, int(coarse * limits[roll_axis]) * 2 - 1))

        for i in tqdm(range(int(coarse * limits[roll_axis])), desc='computing correlation'):
            uu_rolled = np.roll(uu, i, axis=roll_axis)
            x_grid_rolled, y_grid_rolled = np.roll(x_grid, i, axis=roll_axis), np.roll(y_grid, i, axis=roll_axis)
            r_grid = np.sqrt((x_grid_rolled - x_grid) ** 2 + (y_grid_rolled - y_grid) ** 2)
            corr_uu = uu * uu_rolled / uu2_norm  # correlation values
            rr[:, i] = r_grid.flatten()
            corr[:, i] = corr_uu.flatten()

        # flatten arrays to feed to binned_statistic
        rr, corr = rr.flatten(), corr.flatten()

        # get a histogram
        rr_, _, _ = binned_statistic(rr, rr, statistic='mean', bins=n_bins)
        corr_, _, _ = binned_statistic(rr, corr, statistic='mean', bins=n_bins)
        corr_err, _, _ = binned_statistic(rr, corr, statistic='std', bins=n_bins)

        # Sort arrays
        rr, corr = sort2arr(rr_, corr_)
        rr, corr_err = sort2arr(rr_, corr_err)

        # Insert to a big array
        rrs[..., t] = rr
        corrs[..., t] = corr
        corr_errs[..., t] = corr_err

    return rrs, corrs, corr_errs

# def get_two_point_vel_corr_iso(udata):
#     """
#
#     Parameters
#     ----------
#     udata: 5D or 4D numpy array, 5D if the no. of spatial dimensions is 3. 4D if the no. of spatial dimensions is 2.
#           ... (ux, uy, uz) or (ux, uy)
#           ... ui has a shape (height, width, depth, duration) or (height, width, depth) (3D)
#           ... ui may have a shape (height, width, duration) or (height, width) (2D)
#
#
#     Returns
#     -------
#     rij: nd array (r, t, i, j)
#         ... Rij (\vec{r} , t) = <u_j(\vec{x}) u_j(\vec{x}+\vec{r})> / <u_j(\vec{x})^2>
#         ... If system is homogeneous and isotropic,
#                         Rij (\vec{r} , t) = u_rms^2 [g(r,t) delta_ij + {f(r,t) - g(r,t)} r_i * r_j / r^2]
#             where f, and g are long. and transverse autocorrelation functions.
#     """
#     udata = fix_udata_shape(udata)
#     dim = len(udata)
#     if dim == 2:
#         dim, height, width, duration = udata[0].shape
#     elif dim == 3:
#         dim, height, width, depth, duration = udata[0].shape
#     compute_spatial_autocorr




# Sample velocity field
def rankine_vortex_2d(xx, yy, x0=0, y0=0, gamma=1., a=1.):
    """
    Reutrns a 2D velocity field with a single Rankine vortex at (x0, y0)

    Parameters
    ----------
    xx
    yy
    x0
    y0
    gamma
    a

    Returns
    -------
    udata: (ux, uy)

    """
    rr, phi = fa.cart2pol(xx - x0, yy - y0)

    cond = rr < a
    ux, uy = np.empty_like(rr), np.empty_like(rr)

    ux[cond] = -gamma * rr[cond] / (2 * np.pi * a ** 2) * np.sin(phi[cond])
    uy[cond] = gamma * rr[cond] / (2 * np.pi * a ** 2) * np.cos(phi[cond])
    ux[~cond] = -gamma / (2 * np.pi * rr[~cond]) * np.sin(phi[~cond])
    uy[~cond] = gamma / (2 * np.pi * rr[~cond]) * np.cos(phi[~cond])

    udata = np.stack((ux, uy))

    return udata


def rankine_vortex_line_3d(xx, yy, zz, x0=0, y0=0, gamma=1., a=1., uz0=0):
    """
    Reutrns a 3D velocity field with a Rankine vortex filament at (x0, y0, z)

    Parameters
    ----------
    xx: 3d numpy grid
    yy: 3d numpy grid
    zz: 3d numpy grid
    x0: float, location of Rankine vortex filament
    y0: float, location of Rankine vortex filament
    gamma: float, circulation
    a: float, core radius
    uz0: float, constant velocity component in the z-direction

    Returns
    -------
    udata: (ux, uy, uz)

    """
    rr, theta, phi = fa.cart2sph(xx - x0, yy - y0, zz)

    cond = rr < a
    ux, uy, uz = np.empty_like(rr), np.empty_like(rr), np.empty_like(rr)

    ux[cond] = -gamma * rr[cond] / (2 * np.pi * a ** 2) * np.sin(phi[cond])
    uy[cond] = gamma * rr[cond] / (2 * np.pi * a ** 2) * np.cos(phi[cond])
    ux[~cond] = -gamma / (2 * np.pi * rr[~cond]) * np.sin(phi[~cond])
    uy[~cond] = gamma / (2 * np.pi * rr[~cond]) * np.cos(phi[~cond])

    uz = np.ones_like(ux) * uz0

    udata = np.stack((ux, uy, uz))

    return udata


def get_sample_turb_field_3d(return_coord=True):
    mod_loc = os.path.abspath(__file__)
    pdir, filename = os.path.split(mod_loc)
    datapath = os.path.join(pdir, 'sample_data/isoturb_slice2.h5')
    data = rw.read_hdf5(datapath)

    keys = data.keys()
    keys_u = [key for key in keys if 'u' in key]
    keys_u = fa.natural_sort(keys_u)
    duration = len(keys_u)
    depth, height, width, ncomp = data[keys_u[0]].shape
    udata = np.empty((ncomp, height, width, depth, duration))
    for t in range(duration):
        udata_tmp = data[keys_u[t]]
        udata_tmp = np.swapaxes(udata_tmp, 0, 3)
        udata[..., t] = udata_tmp
    data.close()

    if return_coord:
        x, y, z = range(width), range(height), range(depth)
        xx, yy, zz = np.meshgrid(y, x, z)
        return udata, xx, yy, zz
    else:
        return udata

# turbulence related stuff
def get_rescaled_energy_spectrum_saddoughi():
    """
    Returns values to plot rescaled energy spectrum from Saddoughi (1992)
    Returns
    -------

    """
    k = np.asarray([1.27151, 0.554731, 0.21884, 0.139643, 0.0648844, 0.0198547, 0.00558913, 0.00128828, 0.000676395, 0.000254346])
    e = np.asarray([0.00095661, 0.0581971, 2.84666, 11.283, 59.4552, 381.78, 2695.48, 30341.9, 122983, 728530])
    return k, e


# misc
def fix_udata_shape(udata):
    """
    It is better to always have udata with shape (height, width, depth, duration) (3D) or  (height, width, duration) (2D)
    This method fixes the shape of udata such that if the original shape is  (height, width, depth) or (height, width)
    Parameters
    ----------
    udata: nd array,
          ... with shape (height, width, depth) (3D) or  (height, width, duration) (2D)
          ... OR shape (height, width, depth, duration) (3D) or  (height, width, duration) (2D)

    Returns
    -------
    udata: nd array, with shape (height, width, depth, duration) (3D) or  (height, width, duration) (2D)

    """
    shape = udata.shape  # shape=(dim, nrows, ncols, nstacks) if nstacks=0, shape=(dim, nrows, ncols)
    if shape[0] == 2:
        ux, uy = udata[0, ...], udata[1, ...]
        try:
            dim, nrows, ncols, duration = udata.shape
            return udata
        except:
            dim, nrows, ncols = udata.shape
            duration = 1
            ux = ux.reshape((ux.shape[0], ux.shape[1], duration))
            uy = uy.reshape((uy.shape[0], uy.shape[1], duration))
            return np.stack((ux, uy))

    elif shape[0] == 3:
        dim = 3
        ux, uy, uz = udata[0, ...], udata[1, ...], udata[2, ...]
        try:
            # print ux.shape
            nrows, ncols, nstacks, duration = ux.shape
            return udata
        except:
            nrows, ncols, nstacks = ux.shape
            duration = 1
            ux = ux.reshape((ux.shape[0], ux.shape[1], ux.shape[2], duration))
            uy = uy.reshape((uy.shape[0], uy.shape[1], uy.shape[2], duration))
            uz = uz.reshape((uz.shape[0], uz.shape[1], uz.shape[2], duration))
            return np.stack((ux, uy, uz))

def get_equally_spaced_grid(udata, spacing=1):
    """
    Returns a grid to plot udata
    Parameters
    ----------
    udata
    spacing: spacing of the grid

    Returns
    -------
    xx, yy, (zz): 2D or 3D numpy arrays
    """
    dim = len(udata)
    if dim == 2:
        height, width, duration = udata[0].shape
        x, y = range(width), range(height)
        xx, yy = np.meshgrid(x, y)
        return xx * spacing, yy * spacing
    elif dim == 3:
        height, width, depth, duration = udata[0].shape
        x, y, z = range(width), range(height), range(depth)
        xx, yy, zz = np.meshgrid(y, x, z)
        return xx * spacing, yy * spacing, zz * spacing

