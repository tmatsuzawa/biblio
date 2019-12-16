import numpy as np

def fft2d(aa, x0=0, x1=None, y0=0, y1=None,
                           z0=0, z1=None, dx=None, dy=None, dz=None):

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
    energy_fft: nd array with shape (height, width, duration) or (height, width, depth, duration)
    ks: nd array with shape (ncomponents, height, width, duration) or (ncomponents, height, width, depth, duration)
        ...



    Example
    -----------------
    nx, ny = 100, 100
    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(0, 4*np.pi, ny)
    dx, dy = x[1]- x[0], y[1]-y[0]

    # Position grid
    xx, yy = np.meshgrid(x, y)

    # In Fourier space, energy will have a peak at (kx, ky) = (+/- 5, +/- 2)
    ux = np.sin(2.5*xx + yy)
    uy = np.sin(yy) * 0
    udata_test = np.stack((ux, uy))
    ek, ks = vel.get_energy_spectrum_nd(udata_test, dx=dx, dy=dy)
    graph.color_plot(xx, yy, (ux**2 + uy**2 ) /2., fignum=2, subplot=121)
    fig22, ax22, cc22 = graph.color_plot(ks[0], ks[1], ek.real[..., 0], fignum=2, subplot=122, figsize=(16, 8))
    graph.setaxes(ax22, -10, 10, -10, 10)

    # Power amplitude
    Note that power amplitude by FFT is already regularized to satisdy Parseval's theorem.
    1/n_samples * sum of f(k)^2 over k = sum of f(x)^2 over x
    """
    if dx is None or dy is None:
        print 'ERROR: dx or dy is not provided! dx is grid spacing in real space.'
        print '... k grid will be computed based on this spacing! Please provide.'
        raise ValueError
    if x1 is None:
        x1 = aa.shape[1]
    if y1 is None:
        y1 = aa.shape[0]

    dim = len(aa.shape)

    aa = aa[y0:y1, x0:x1]


    n_samples = 1
    for i in range(len(aa.shape)):
        n_samples *= aa.shape[i]
    aa_fft = np.abs(np.fft.fftn(aa))
    aa_fft = np.fft.fftshift(aa_fft)
    # aa_fft /= np.sqrt(n_samples)


    if dim == 2:
        height, width = aa.shape
        kx = np.fft.fftfreq(width, d=dx)  # this returns FREQUENCY (JUST INVERSE LENGTH) not ANGULAR FREQUENCY
        ky = np.fft.fftfreq(height, d=dy)
        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)
        kxx, kyy = np.meshgrid(kx, ky)
        kxx, kyy = kxx * 2 * np.pi, kyy * 2 * np.pi # Convert inverse length into wavenumber

        return aa_fft, np.asarray([kxx, kyy])

def get_kgrid_2d(aa, dx=None, dy=None):
    height, width = aa.shape
    kx = np.fft.fftfreq(width, d=dx)  # this returns FREQUENCY (JUST INVERSE LENGTH) not ANGULAR FREQUENCY
    ky = np.fft.fftfreq(height, d=dy)
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    kxx, kyy = np.meshgrid(kx, ky)
    kxx, kyy = kxx * 2 * np.pi, kyy * 2 * np.pi # Convert inverse length into wavenumber
    return kxx, kyy