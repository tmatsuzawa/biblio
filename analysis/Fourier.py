# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:34:20 2015

@author: stephane
"""

import numpy as np
import time
import pylab as plt
import turbulence.analysis.cdata as cdata
import turbulence.display.graphes as graphes
import turbulence.analysis.basics as basics
import turbulence.analysis.vgradient as vgradient
import turbulence.manager.access as access
import sys

'''Module containing functions for taking fourier transforms of data, typically 3d data of 2d flow fields over time.
'''


def movie_spectrum(M, field, alpha=[-5. / 3], Dt=10, fignum=1, start=0, stop=0):
    # switch_field(field)

    if not hasattr(M, field):
        M, field = vgradient.compute(M, field)

    if not hasattr(M, 'S_' + field):
        yy = getattr(M, field)
        yy_k, k = spectrum_1d(yy, M, display=False, Dt=Dt)
        print(yy_k.shape)
    else:
        yy_k = getattr(M, 'S_' + field)
        k = getattr(M, 'k_' + field)
    step = max(Dt / 2, 1)
    N, nt = yy_k.shape

    if stop == 0:
        tax = range(start, nt, step)
    else:
        tax = range(start, stop, step)

    figs = {}

    for i, t in enumerate(tax):
        # graphes.cla(fignum)
        graphes.graph(k, yy_k[:, t], label='k-', fignum=fignum)
        # graphes.graphloglog(k,yy_k[:,t],label='k-',fignum=fignum)
        add_theory(k, yy_k[:, t], alpha, fignum=fignum)
        # graphes.set_axis(10**-2,10**0,10**-4,10**2)
        figs.update(graphes.legende('k (mm^-1)', 'E_k (mm^3/s^-2)', ''))

        graphes.save_graphes(M, figs, prefix='Movie_Spectrum_' + field + '/', suffix='_' + str(t))


def add_theory(k, yy_k, alpha, fignum=0):
    for a in alpha:
        k0 = np.nanmean(k)
        val = np.nanmean(yy_k * (k / k0) ** (-a))
        std_val = np.nanstd(yy_k * (k / k0) ** (-a))

        #  print('Corrected spectrum : '+str(std_val/val*100)+' %')
        graphes.graphloglog(k, val * (k / k0) ** a, label='r--', fignum=fignum)


def switch_field(M, field):
    """

    M:
    field:

    Returns
    -------
    """
    theory = {}
    theory['E'] = {'variable': M.E, 'exponent': -5. / 3, 'Unit': 'mm^3/s^-2', 'name': 'Energy'}
    theory['Enstrophy'] = {'variable': M.omega ** 2, 'exponent': -1, 'Unit': 'mm/s^-2', 'name': 'Enstrophy'}
    theory['Strain'] = {'variable': M.strain, 'exponent': -1, 'Unit': 'mm/s^-2', 'name': 'Strain'}
    return theory


def compute_spectrum_2d(M, Dt=10):
    S_E, kx, ky = energy_spectrum_2d(M, display=False, Dt=Dt)
    setattr(M, 'kx', kx)
    setattr(M, 'ky', ky)
    setattr(M, 'S_E', S_E)
    return S_E, kx, ky


def compute_spectrum_1d(mm, Dt=10):
    """

    Parameters
    ----------
    M :
    Dt :

    Returns
    -------
    """
    S_k, k = energy_spectrum_1d(mm, display=False, Dt=Dt)
    setattr(mm, 'k', k)
    setattr(mm, 'S_k', S_k)
    return S_k, k


def compute_spectrum_1d_within_region(mm, radius=None, polygon=None, display=False, dt=10, center_frame=False):
    """description

    Parameters
    ----------
    mm : Mdata_PIVlab class instance
        The dataset on which to compute the 1d spectrum
    radius : float or None
        the radius of a disc to examine, if not None. If region is not None, radius is ignored
    polygon : #vertices x 2 numpy float array
        If not none, use this closed path to define the region of interes
    display : bool
        show some intermediate results to check that code is functioning
    dt : int (default=10)
        Number of frames to smooth over
    center_frame : bool
        Cut off the edges of the frame, keeping the center within distance 'radius' of center pixel rather than using
        mm.x and mm.y for position information

    Returns
    -------
    s_k :
    k :
    """
    import turbulence.analysis.data_handling as dh
    data = access.get_all(mm, 'E')
    if polygon is not None:
        # Use matplotlib to find points in path
        dh.pts_in_polygon(mm.x, mm.y, polygon=polygon)
    elif radius is not None:

        # # Circular mask
        # mmr = (mm.x - np.nanmean(mm.x)) ** 2 + (mm.y - np.nanmean(mm.y)) ** 2
        # include = (np.abs(mmr) < radius).astype(np.int)
        # mdatr = data * include[:, :, np.newaxis]
        #
        # # Check it
        # if display:
        #     import turbulence.display.plotting as tplt
        #     toview = [0, 20]
        #     for ind in toview:
        #         frame = mdatr[:, :, ind]
        #         print 'np.shape(frame) = ', np.shape(frame)
        #         tplt.plot_real_matrix(frame, show=True, close=True)

        if center_frame:
            cols = np.unique(np.where(np.abs(mm.x - np.nanmean(mm.x)) < radius)[1])
            rows = np.unique(np.where(np.abs(mm.y - np.nanmean(mm.y)) < radius)[0])
            # include = np.logical_and(np.abs(mm.x - np.nanmean(mm.x)) < radius,
            #                          np.abs(mm.y - np.nanmean(mm.y)) < radius)
        else:
            cols = np.unique(np.where(np.abs(mm.x) < radius)[1])
            rows = np.unique(np.where(np.abs(mm.y) < radius)[0])
            # include = np.logical_and(np.abs(mm.x) < radius, np.abs(mm.y) < radius)

        mdatr = data[np.min(rows):np.max(rows), np.min(cols):np.max(cols), :]

        # print 'include = ', include
        # print 'np.shape(include) = ', np.shape(include)

        if display:
            # print 'mm.y = ', mm.y
            # print 'radius = ', radius
            # print 'rows = ', rows
            # print 'cols = ', cols
            print 'np.shape(data) = ', np.shape(data)
            print 'np.shape(mdatr) = ', np.shape(mdatr)
            import turbulence.display.plotting as tplt
            toview = [0, 20]
            for ind in toview:
                frame = mdatr[:, :, ind]
                print 'np.shape(frame) = ', np.shape(frame)
                tplt.plot_real_matrix(frame, show=True, close=True)

    else:
        raise RuntimeError('Must supply either radius or region values to compute_spectrum_1d_within_region()')

    # use M.Ux
    # test: see turbulence/scripts/fourier_shells_test.py
    dx = np.mean(np.diff(mm.x))
    s_e, kx, ky = spectrum_2d(mdatr, M=None, dx=dx, Dt=dt, display=display)
    nk_approx = min(np.shape(mdatr)[0:2])
    s_k, k = spectrum_2d_to_1d_convert(s_e, kx, ky, dt=dt, nkout=nk_approx + 1)
    if display:
        print('kx = ', np.shape(kx))
        print('ky = ', np.shape(ky))
        print('s_e = ', np.shape(s_e))
        # print('k = ', k)
        # print('s_k = ', s_k)
        plt.plot(k, s_k, 'b.-')
        plt.show()

    return s_k, k


def energy_spectrum_2d(mm, display=False, field='E', Dt=10):
    """Compute the 2 dimensionnal energy spectrum of a Mdata class instance

    Parameters
    ----------
    mm : Mdata class instance, or any other object that contains the following fiels :
        methods : shape()
        attributes : Ux, Uy
    display : bool. Default False
        display resulting spectrum
    Dt : int. Default value 10
        time window to smooth the data with turbulence.analysis.basics.smooth


    Returns
    -------
    S_E : 3d np array
        Power spectrum of Ux and Uy velocity components
    kx : 2d np array
        wave-vector along x
    ky : 2d np array
        wave-vector along y
    """
    data = access.get_all(mm, field)
    S_E, kx, ky = spectrum_2d(data, mm, Dt=Dt)
    return S_E, kx, ky


def spectrum_2d(yy, M=None, dx=None, Dt=5, display=False, display_frames=None):
    """
    Compute 2d spatial spectrum of yy. If a Mdata object is specified, use the spatial scale of M.x
    to scale the spectrum

    Parameters
    -----
    yy : 3d numpy array
        Compute the spectrum along the first two dimensions
    M : Mdata class instance or None
        If supplied, M.x is used to supply an absolute scale for the k vectors.
    dx : float
        If M is not supplied,
    Dt : int
        points over which to smooth the spectrum?
    display : bool
        show the resulting spectrum. Note that if the object yy is 3d, we display the frames in display_frames, one by
        one, or only the first frame if display_frames is None
    display_frames : bool or None
        the (time) slices of the spectrum of yy to display if yy is 3d. If None and display==True, display just the
        first frame
    """
    # cropping for the 2016_08_03
    #    yy = yy[:,5:]

    nx, ny, nt = yy.shape
    # kx=np.arange(-(nx-1)/2,(nx-1)/2+1,1)
    # ky=np.arange(-(ny-1)/2,(ny-1)/2+1,1)
    kx, ky = np.mgrid[-nx / 2:nx / 2:complex(0, nx), -ny / 2:ny / 2:complex(0, ny)]

    # distance between two measure in mm
    if M is not None:
        dx = np.mean(np.diff(M.x))
        #    print('dx : ' +str(dx))
        if dx == 0:
            dx = 1
    elif dx is None:
        dx = 1
    # Note that otherwise dx = supplied dx

    kx /= (dx * nx)
    ky /= (dx * ny)  # in mm^-1
    #   print(np.shape(E))
    # yy=basics.smooth(yy,Dt)

    result = cdata.rm_nans([yy])
    # print 'Fourier: result = ', result
    vel3d = result[0]  # cdata.rm_nans([E])

    #    print(np.where(np.isnan(E)))
    # S_E = np.zeros(np.shape(yy))
    s_e = np.abs(np.fft.fftn(vel3d, axes=(0, 1))) * dx ** 2 / (nx * ny)
    s_e = np.fft.fftshift(s_e, axes=(0, 1))

    # smooth the spectrum by averaging over Dt time steps in time
    s_e = basics.smooth(s_e, Dt / 2)

    if display:
        # check if 3d. If so, display only certain frames decided by display_frames
        if len(np.shape(s_e)) > 2:
            if display_frames is None:
                display_frames = [0]
            for frame in display_frames:
                display_fft_2d(kx, ky, s_e[:, :, frame], fignum=1, vmin=0, vmax=7)
        else:
            display_fft_2d(kx, ky, s_e, fignum=1, vmin=0, vmax=7)
    return s_e, kx, ky


def spectrum_1d(yy, M=None, display=False, Dt=5):
    """Compute the 1 dimensionnal energy spectrum of yy
    The computation is done by averaging over circles in k space from a 2d spectrum

    Parameters
    ----------
    m : Mdata class instance, or any other object that contains the following fields :
        methods : shape()
        attributes : Ux, Uy
    display : bool. Default False
        display resulting spectrum
    Dt : int. Default value 10
        time window to smooth the data with turbulence.analysis.basics.smooth

    Returns
    -------
    S_k : 2d np array
        1d Power spectrum of Ux and Uy velocity components
    kbin : 1d np array
        wave-vector
    """
    # compute the fft 2d, then divide in slices of [k,k+dk]
    print('Fourier.spectrum_1d(): Compute 2d fft')
    S_E, kx, ky = spectrum_2d(yy, M, display=False, Dt=Dt)
    nx, ny, nt = np.shape(S_E)

    k = np.sqrt(np.reshape(kx ** 2 + ky ** 2, (nx * ny,)))
    # note: stephane was missing the squaring operation below
    kx_1d = np.sqrt(np.reshape(kx ** 2, (nx * ny,)))
    ky_1d = np.sqrt(np.reshape(ky ** 2, (nx * ny,)))

    S_E = np.reshape(S_E, (nx * ny, nt))
    # sort k by values
    #    indices=np.argsort(k)
    #    S_E=S_E[indices]

    Nbit = int(min(nx, ny) * 2.)
    nk, nbin = np.histogram(k, Nbit)
    N = len(nbin) - 1

    # remove too small values of kx or ky (components aligned with the mesh directions)
    epsilon = nbin[4]
    kbin = np.zeros(N + 1)
    indices = np.zeros((nx * ny, N), dtype=bool)
    for i in range(N):
        indices[:, i] = np.logical_and(np.logical_and(k >= nbin[i], k < nbin[i + 1]),
                                       np.logical_and(np.abs(kx_1d) >= epsilon, np.abs(ky_1d) >= epsilon))
        kbin[i] = np.mean(k[indices[:, i]])
    kbin[N] = 2 * kbin[N - 1] - kbin[N - 2]
    #   print(len(indices[i]))
    #    kbin=(np.cumsum(nbin)[2:]-np.cumsum(nbin)[0:-2])/2

    S_k = np.zeros((N, nt))

    if M is not None:
        dx = np.mean(np.diff(M.x))
        if dx == 0:
            dx = 1
    else:
        dx = 1

    print('Fourier.spectrum_1d(): Compute 1d fft from 2d')
    for t in range(nt):
        for i in range(N):
            S_part = S_E[:, t]
            dk = kbin[i + 1] - kbin[i]
            # print(dk)
            S_k[i, t] = np.nanmean(S_part[indices[:, i]]) * np.sqrt(nx * ny) / dx  # 2*np.pi*dk  #spectral surface
            #    print(S_k[i,t])

    # averaged in time ??
    S_k = basics.smooth(S_k, Dt)
    print('Fourier.spectrum_1d(): Done')

    #    print(len(np.where(np.isnan(S_k))[0]))
    if display:
        plt.plot(kbin[:-1], S_k)
        plt.show()
        plt.clf()

    return S_k, kbin[:-1]


def energy_spectrum_1d(M, display=False, Dt=10):
    """Compute the 1 dimensionnal energy spectrum of a Mdata class instance
    The computation is done by averaging over circles in k space from a 2d spectrum

    Parameters
    ----------
    m : Mdata class instance, or any other object that contains the following fields :
        methods : shape()
        attributes : Ux, Uy
    display : bool. Default False
        display resulting spectrum
    Dt : int (defaul=10)
        time window to smooth the data with turbulence.analysis.basics.smooth

    Returns
    -------
    S_k : 2d np array
        1d Power spectrum of Ux and Uy velocity components
    kbin : 1d np array
        wave-vector
    """
    # compute the fft 2d, then divide in slices of [k,k+dk]
    print('Fourier.energy_spectrum_1d(): Compute 2d fft')
    s_e, kx, ky = energy_spectrum_2d(M, display=False, Dt=Dt)
    S_k, kbin = spectrum_2d_to_1d_convert(s_e, kx, ky, dt=Dt)

    return S_k, kbin


def spectrum_2d_to_1d_convert(s_e, kx, ky, dt=10, nkout=30, verbose=False):
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

    Returns
    -------
    s_k :
    kbin :
    """
    nx, ny, nt = np.shape(s_e)

    kk = np.sqrt(np.reshape(kx ** 2 + ky ** 2, (nx * ny,)))
    kx_1d = np.sqrt(np.reshape(kx ** 2, (nx * ny,)))
    ky_1d = np.sqrt(np.reshape(ky ** 2, (nx * ny,)))

    s_e = np.reshape(s_e, (nx * ny, nt))
    # sort k by values
    #    indices=np.argsort(k)
    #    s_e=s_e[indices]

    nk, nbin = np.histogram(kk, bins=nkout)
    print 'Fourier.spectrum_2d_to_1d_convert(): nkout = ', nkout
    print 'Fourier.spectrum_2d_to_1d_convert(): nbin = ', nbin
    nn = len(nbin) - 1

    # remove too small values of kx or ky (components aligned with the mesh directions)
    epsilon = nbin[2]
    kbin = np.zeros(nn)
    indices = np.zeros((nx * ny, nn), dtype=bool)
    jj = 0
    okinds_nn = []
    for ii in range(nn):
        print 'ii = ', ii
        indices[:, ii] = np.logical_and(np.logical_and(kk >= nbin[ii], kk < nbin[ii + 1]),
                                        np.logical_and(np.abs(kx_1d) >= epsilon, np.abs(ky_1d) >= epsilon))

        # If there are any values to add, add them to kbin (prevents adding values less than epsilon
        if indices[:, ii].any():
            kbin[jj] = np.mean(kk[indices[:, ii]])
            okinds_nn.append(ii)
            jj += 1

        # Check for an error
        # print 'kbin[', ii, '] = ', kbin[ii]
        # print 'kbin[', ii, '] = ', kk[indices[:, ii]]
        # print 'indices[:, ii] = ', indices[:, ii]
        # print 'kk = ', kk
        # print 'kx_1d = ', np.abs(kx_1d)
        # print 'ky_1d = ', np.abs(ky_1d)
        # print 'epsilon = ', epsilon
        # sys.exit()

    print 'Fourier.spectrum_2d_to_1d_convert(): np.shape(kbin) = ', np.shape(kbin)
    s_k = np.zeros((nn, nt))
    # s_part = np.zeros(nx * ny)

    print('Fourier.spectrum_2d_to_1d_convert(): Compute 1d fft from 2d')
    for t in range(nt):
        s_part = s_e[:, t]
        jj = 0
        for ii in okinds_nn:
            s_k[jj, t] = np.nanmean(s_part[indices[:, ii]])
            jj += 1
            # print(S_k[i,t])

    # averaged in time ??
    s_k = basics.smooth(s_k, dt)

    print('Fourier.spectrum_2d_to_1d_convert(): np.shape(kbin) = ', np.shape(kbin))
    print('Fourier.spectrum_2d_to_1d_convert(): np.shape(s_k) = ', np.shape(s_k))
    return s_k, kbin


def display_fft_vs_t(m, dimension='1d', Dt=20, fignum=0, label='^', display=False):
    """
    Parameters
    ----------
    m :
    dimension:
    Dt:
    fignum:
    label:
    display:

    Returns
    -------
    t, E_t
    """
    display_part = True
    #  plt.close(1)
    if dimension == '1d':
        S_k_2d, kx, ky = energy_spectrum_2d(m, Dt=Dt)
        S_k, k = energy_spectrum_1d(m, Dt=Dt)
    if dimension == '2d':
        S_k, kx, ky = energy_spectrum_2d(m, Dt=Dt)
    # start=580
    #    end=700
    #    step=10
    # print(S_k)
    if dimension == '1d':
        x = [10 ** 0, 10 ** 1.5]
        y = [10 ** -0.5, 10 ** -3]
        # graphes.graph(x,y,-1,'r-')

        #   t0=590
        # time_serie=range(t0+10,10000,50)#[round(t0*i**2) for i in np.arange(1,4,0.3)]

    # origin of time
    t0 = 0.  # .51
    tref = np.asarray(m.t) - t0
    #  tref=1-tref
    nt = len(tref)
    # time_serie=[600,900,1200,1600,1900,2900,5000,8000]

    #  time_serie=[i for i in np.arange(400,650,10)]
    #  time_serie=range(10,nt-2)
    step = 1
    time_serie = range(Dt + 1, nt - Dt * 3 - 11, step)  # [50,120,200,300,400,450,550]
    # print(nt-Dt*3-11)
    #   t0=500
    #   time_serie=[round(i)+t0 for i in np.logspace(1,3.973) if round(i)+t0<nt]
    # time_serie=range(start,end,50)

    alpha = np.zeros(len(time_serie))
    beta = np.zeros(len(time_serie))
    epsilon = np.zeros(len(time_serie))

    t_alpha = np.zeros(len(time_serie))

    #    graphes.hist(k)
    kmin = -2.7
    kmax = -1.7
    # print(np.log10(k))
    tmax = 300;
    for i, t in enumerate(time_serie):
        #   print(t)
        if tref[t] < tmax:
            if dimension == '1d':
                k_log = np.log10(k)
                S_log = np.log10(S_k[:, t])
                indices = np.logical_and(k_log > kmin, k_log < kmax)
                #  print(indices)
                #  print(S_log)
                k_log = k_log[indices]
                S_log = S_log[indices]
                P = np.polyfit(k_log, S_log, 1)

                alpha[i] = 10 ** P[1]  # *np.mean(k)**P[0]
                beta[i] = P[0]

                C_k = 0.55
                epsilon[i] = (alpha[i] / C_k) ** (3 / 2)
                t_alpha[i] = tref[t]

                # if t>min(time_serie):
                #     Dt=tref[time_serie.index(t)]-tref[time_serie.index(t-1)]
                #     print(Dt,alpha[time_serie.index(t)])
                #    print((t_alpha,alpha))
                if display_part:
                    graphes.set_fig(1)
                    # graphes.subplot(1,2,1)
                    k0 = np.min(k);
                    display_fft_1d(k, (k / k0) ** (5 / 3) * S_k[:, t] / alpha[i], fignum=1, label='')
                    display_fft_1d(k, (k / k0) * S_k[:, t] / alpha[i], fignum=2, label='')

                    # normalized
                    #   print(t_alpha[i])
                    # display_fft_1d(k,np.abs(S_k[:,t]/t_alpha[i]),fignum=1)

                    # graphes.graphloglog(k[indices],10**np.polyval(P,k_log),label='r--')
                    display_fft(m, t, dimension)

                    # graphes.subplot(1,2,2)
                    #  graphes.vfield_plot(m,t,fignum=2)

                    # there is a slighlty degeneracy of the spectrum along both axis. Removes |kx| < epsilon and |ky| < epsilon for every k ?
                    #   display_fft_2d(kx,ky,S_k_2d[:,:,t],fignum=3)
                    #   display_fft(m,t,dimension)


                    #   input()

            if dimension == '2d':
                display_fft_2d(kx, ky, S_k[:, :, t])
                display_fft(m, t, dimension)

    if display:
        #        title='$Z$ = '+str(m.Sdata.param.Zplane/10)+' cm'
        #    graphes.legende('$t$ (s)','$E (a.u.)$',title
        graphes.graphloglog(t_alpha, alpha, label=label, fignum=7)
        graphes.graphloglog([10 ** -1, 10 ** 3], [10 ** 8, 10 ** 0], label='r--', fignum=7)
        graphes.legende('$t$ (s)', '$E_{\lambda}$ (a.u.)', graphes.set_title(m))

        graphes.semilogx(t_alpha, beta, label=label, fignum=8)
        #    graphes.semilogx(t_alpha,beta,label=label,fignum=0)
        graphes.semilogx([10 ** -1, 10 ** 3], [-5 / 3, -5 / 3], label='r-', fignum=8)
        graphes.set_axis(10 ** -1, 10 ** 3, -2.5, 0)
        graphes.legende('$t$ (s)', 'exponent', graphes.set_title(m))

        # plot the dissipative scale as a function of time
        nu = 1  # in mm^2/s
        eta = (nu ** 3 / np.asarray(epsilon)) ** (1 / 4)
        graphes.graphloglog(t_alpha, eta, label=label, fignum=9)
        graphes.graphloglog([10 ** -1, 10 ** 3], [10 ** 8, 10 ** 0], label='r--', fignum=9)
        graphes.legende('$t$ (s)', '$\eta$ (mm)', graphes.set_title(m))

    E_t = epsilon
    t = t_alpha
    return t, E_t
    #  graphes.graphloglog([10**2,10**3],[2*10**5,2*10**3],label='r-',fignum=2)


def display_fft(m, i, tag):
    # to be replaced by m.z
    if hasattr(m.Sdata.param, 'Zplane'):
        Z = m.Sdata.param.Zplane / 10
    else:
        Z = -10
    title = '$Z$ = ' + str(Z) + ' cm, $t$ = ' + str(m.t[i]) + ' ms'

    Dir = m.fileDir + 'FFT_vs_t_part_' + tag + '_' + m.id.get_id() + '/'

    if tag == '1d':
        graphes.legende('$k$ (mm$^{-1}$)', '$E_k$ (a.u.)', title)
    if tag == '2d':
        graphes.legende('$k_x$ (mm$^{-1}$)', '$k_y$ (mm$^{-1}$)', title)

    return None


def display_fft_2d(kx, ky, sk, fignum=1, vmin=1, vmax=7, log10=True):
    """Preview the 2d spectrum (typically energy spectrum) in log scale

    Parameters
    ----------
    kx : n x m float array
        wavenumbers in x dimension
    ky : n x m float array
        wavenumbers in y dimension
    S : n x m complex? array
        the fourier transform to display, supplied in raw linear scale
    fignum : int
        The index of the figure on which to plot the fft
    vmin : float
        minimum for the colorscale
    vmax : float
        maximum for the colorscale

    Returns
    -------
    cc :
    """
    # display in logscale
    fig, ax, cc = graphes.color_plot(kx, ky, sk, log10=log10, vmin=vmin, vmax=vmax, fignum=fignum)

    graphes.clegende(cc, '$\ln (E_k) [a.u.]$')
    return cc


def display_fft_1d(k, S, fignum=1, label='k^', vmin=-3, vmax=0.5, theory=False, alpha=-5. / 3):
    # display in logscale
    #    k_log=np.log10(k)
    #    S_log=np.log10(S)
    #    print(k)
    #    print(S)
    graphes.graphloglog(k, S, fignum=fignum, label=label)

    if theory:
        A = np.mean(S * k ** (-alpha))
        graphes.graphloglog(k, A * k ** alpha, label='r--', fignum=fignum)

        #   plt.figure(2)
        #   plt.hist(np.reshape(S_i,(nx*ny,1)),50)
        #   plt.show(False)
        #   plt.pause(5)
        #   print(np.shape(S_i))


def main():
    #    for M in Mlist[:
    #  display_fft_vs_t(Mlist[4],'1d',Dt=100)
    #    display_fft_vs_t(Mlist[3],'1d',Dt=10)
    #    display_fft_vs_t(Mlist[2],'1d',Dt=10)
    #  display_fft_vs_t(Mlist2[4],'1d',Dt=50)

    #    display_fft_vs_t(Mlist2[0],'1d',Dt=50)

    indices = range(5)  # [0,1,2,4]
    labels = ['k^', 'ro', 'bp', 'c8', 'g*']

    for i, indice in enumerate(indices):
        label = labels[i]

        M = Mlist2[indice]

        t, E_t = display_fft_vs_t(M, '1d', Dt=20, label=label)
        #
        #    for M in M_log:
        #      energy_spectrum_2d(M,True)
        #        display_fft_vs_t(M,'1d')
        #    date='2015_03_21'
        #    indexList=[0,2,4,5,6,7,8,9,13,14,16,17]
        #    mindex=0
        #    for index in indexList:
        #        print(index)
        #        M=M_manip.load_Mdata(date,index,mindex)
        #        display_fft_vs_t(M,600,10000,'1d')
        #  display_fft_vs_t(M5,600,5900,'1d','b+')   #-40
        #   display_fft_vs_t(M0,600,10000,'2d','k^')   #-100
        #
        #   display_fft_vs_t(M4,600,2560,'1d','ro') #-60
        #   display_fft_vs_t(M7,600,6000,'1d','kx') #0

        # main()
