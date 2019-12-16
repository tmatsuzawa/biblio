import numpy as np
# from scipy import ndimage,ndarray,interpolate
from matplotlib import backend_bases as bck
from math import *
import pylab as plt
import time
import random
from scipy import ndimage
import os.path

import turbulence.display.graphes as graphes
import turbulence.mdata.Smath as Smath
import turbulence.mdata.Sdata


def multi_mean_profile(Slist, i, j, direction='v', shift=0):
    label = ['k^', 'ro', 'bx', 'mo', 'gx', 'k+']
    for S in Slist:
        plt.clf()
        mean_profile(S, i, j, direction, label[(Slist.index(S) + shift) % len(label)], True)


# mean_profile(S,label[Slist.index(S)%len(label)])

def fix_PIV(S):
    # replace by NaN all the values of Ux and Uy that does not correspond to a bounded velocity
    Umax, Umin = bounded_velocity(S)

    U = np.sqrt(S.Ux ** 2 + S.Uy ** 2)

    Vtrue = np.logical_and(U > Umin, U < Umax)

    Ux = np.where(Vtrue, S.Ux, np.NaN)
    Uy = np.where(Vtrue, S.Uy, np.NaN)

    Ntrue = sum(sum(sum(np.where(Vtrue, 1, 0))))
    Nfalse = sum(sum(sum(np.where(Vtrue, 0, 1))))
    N = Ntrue + Nfalse

    print(('Percentage of removed data points : ' + str((Nfalse * 100) / N) + ' %'))

    return Ux, Uy

    # not NaN elements can be acceded from
    np.median(a[np.invert(np.isnan(a))])


def smooth(V, Dt):
    dim = np.shape(V)
    nt = dim[-1]
    N = len(dim)

    Vs = np.array([np.nanmean(V[..., k - Dt:k + Dt], N - 1) for k in range(Dt, nt - Dt)])
    # some elements are lost during the smoothing operation : we should add the initial and enn elements
    #   Vs=np.array([V[...,:Dt]+[np.nanmean(V[...,k-Dt:k+Dt],N-1) for k in range(Dt,nt-Dt)])

    #    tup=
    order = np.arange(1, N + 1)
    order[N - 1] = 0
    ind = tuple(order)
    Vs = np.transpose(Vs, ind)
    #    indices=[slice(dim[i]) for i in range(N-1)]
    #   indices[N-1]=
    #    for k in range(Dt,nt-Dt):
    #       Vs.append())

    #    if len(dim)==1:
    #        Vs=np.array([np.mean(V[k-Dt:k+Dt]) for k in range(Dt,nt-Dt)])
    #    if len(dim)==2:
    #       Vs=np.array([np.mean(V[:,k-Dt:k+Dt],N-1) for k in range(Dt,nt-Dt)])
    #    if len(dim)==3:
    #        Vs=np.array([np.mean(V[:,:,k-Dt:k+Dt],2) for k in range(Dt,nt-Dt)])
    #    if len(dim)==4:
    #        Vs=np.array([np.mean(V[:,:,:,k-Dt:k+Dt],3) for k in range(Dt,nt-Dt)])
    return Vs


def divergence(S, xlines, ylines, display=False):
    # xlines list of tuple ??
    # take two points (symmetric in respect to X=0?), and mesure the relative distance between each
    # as a function of time
    # relative means in respect to the spatial averaged velocity at the time
    t = S.t
    Dt = 10
    ny, nx, nt = S.shape()

    U_moy, U_std = mean_profile(S, label='k^', display=False)

    n = len(xlines)
    Ds = np.zeros((ny, nx, nt - 2 * Dt))  # first element is centered !

    for i in ylines:
        for pair in xlines:
            dx = S.Ux[i, pair[0], :] + S.Ux[i, pair[1], :]  # antisymetric x profile ?
            dy = S.Uy[i, pair[0], :] - S.Uy[i, pair[1], :]
            D, phi = Smath.cart2pol(dx, dy)

            # Divide by the global average velocity at each time !!
            Ds[ylines.index(i), pair[0], :] = smooth(D, Dt)  # =smooth(D/U_moy,Dt)
            Ds[ylines.index(i), pair[1], :] = Ds[i, pair[0], :]

            if display:
                graphes.graph(t[Dt:-Dt], Ds)
                graphes.legende('t (ms)', 'V (m/s)', 'Velocity difference between two symetric points')
    return Ds


def fluctuations(S, Dx, Dy, Dt):
    # compute the standard deviation from a volume (Dx,Dy,Dt) around each point.
    # Return an array of same size than Vec, containing the local standard deviation
    U = S.Ux ** 2 + S.Uy ** 2
    Ufilt2 = ndimage.filters.gaussian_filter(U, [Dx, Dy, Dt])

    Z = U - Ufilt2
    start = 0
    stop = 1500
    Dirname = 'Velocity_fluctations'
    make_2dmovie(S, Z, start, stop, Dirname)


def horizontal_profile(S, ylines, Dt, start=0):
    nx, ny, nt = S.shape()

    x = S.x[0, :]
    for i in range(start, nt, Dt):
        Ux = np.mean(np.mean(S.Ux[ylines, :, i:i + Dt], axis=0), axis=1)
        Uy = np.mean(np.mean(S.Uy[ylines, :, i:i + Dt], axis=0), axis=1)

        std_Ux = np.std(np.std(S.Ux[ylines, :, i:i + Dt], axis=0), axis=1)
        std_Uy = np.std(np.std(S.Uy[ylines, :, i:i + Dt], axis=0), axis=1)

        plt.subplot(121)
        graphes.graph(x, Ux, 0, std_Ux)
        graphes.legende('x (m)', 'V (m/s)', 'Ux')

        plt.subplot(122)
        graphes.graph(x, Uy, 0, std_Uy)
        graphes.legende('x (m)', 'V (m/s)', 'Uy')

        plt.draw()
        input()


def vertical_profile(S, xlines, Dt, start=0):
    nx, ny, nt = S.shape()

    y = S.y[:, 0]
    for i in range(start, nt, Dt):
        Ux = np.mean(np.mean(S.Ux[:, xlines, i:i + Dt], axis=1), axis=1)
        Uy = np.mean(np.mean(S.Uy[:, xlines, i:i + Dt], axis=1), axis=1)

        # standard deviation computation
        std_Ux = np.sqrt(np.mean(np.mean(abs(S.Ux[:, xlines, i:i + Dt] - Ux) ** 2, axis=1), axis=1))
        std_Uy = np.sqrt(np.mean(np.mean(abs(S.Uy[:, xlines, i:i + Dt] - Uy) ** 2, axis=1), axis=1))

        print(std_Ux)

        plt.subplot(121)
        graphes.graph(y, Ux, std_Ux)
        graphes.legende('z (m)', 'V (m/s)', 'Ux')

        plt.subplot(122)
        graphes.graph(y, Uy, std_Uy)
        graphes.legende('z (m)', 'V (m/s)', 'Uy')

        plt.draw()
        input()


def compare_profil(S1, S2):
    # S1 and S2 must be the same length, and the same x and y dimensions
    # compare velocity measurement obtained with two different frame rate
    # norm and direction
    Ux1, Uy1 = fix_PIV(S1)
    Ux2, Uy2 = fix_PIV(S2)

    U1, theta1 = Smath.cart2pol(Ux1, Uy1)
    U2, theta2 = Smath.cart2pol(Ux2, Uy2)
    print((U2.shape))

    nx, ny, nt = S1.shape()
    indices1, indices2, nt = Sdata.subset_index(S1, S2)
    # locate a subset of identical im_index
    # indices correspond of the number of the U rows, not to the im_index valeurs
    #  indices,nt=subset_index(S1,S2)

    U1 = np.reshape(U1[:, :, indices1], nx * ny * nt)
    U2 = np.reshape(U2[:, :, indices2], nx * ny * nt)

    # if U1 and U2 have not the same length,
    # a fraction can be selected using the attr im_index of each one (to use the same list of images)
    # random extract
    N = 2000
    ind = random.sample(range(nx * ny * nt), N)

    graphes.graph(U1[ind], U2[ind])
    xlabel = 'V (m/s)  at ' + str(S1.timescale * S1.fps) + ' fps'
    ylabel = 'V (m/s)  at ' + str(S2.timescale * S2.fps) + ' fps'

    graphes.legende(xlabel, ylabel, '')

    bounded_velocity(S1, True, 5, 'v')
    bounded_velocity(S2, True, 5, 'h')

    input()


def mean_profile(S, i, j, direction='v', label='k^', display=False):
    # mean profile along the whole field : average on one direction only ! (and small windows on the other direction ?)
    nx, ny, nt = S.shape()

    Ux = S.m.Ux
    Uy = S.m.Uy
    # remove the data out of the PIV bounds
    #    Ux,Uy=fix_PIV(S)

    U = np.sqrt(Ux ** 2 + Uy ** 2)

    #    V=np.reshape(U,(nx*ny,nt))
    # median is not so affected by peak values, but standard deviation definetely !
    # histogramm between vmin and vmax, and remove values out of bound (set to NaN)
    U_moy = []
    U_std = []
    t = S.m.t

    # t=t[0:-1]#-500

    Dt = 2
    if direction == 'v':
        # average along the horizontal direction
        U_moy = [np.mean(np.mean(U[j - Dt:j + Dt, :, k], axis=0), axis=0) for k in range(nt)]
        print('horizontal average')
    else:
        # average along the vertical direction
        U_moy = [np.mean(np.mean(U[:, i - Dt:i + Dt, k], axis=0), axis=0) for k in range(nt)]
        print('vertical average')

    print((np.shape(U_moy)))
    if display:
        # U_moy=np.mean(V[np.invert(np.isnan(V))],axis=0)
        print(('Number of frames : ' + str(len(S.m.t))))

        graphes.graph(t, U_moy, label)
        graphes.legende('t (ms)', '<V>_{x,y} (m/s)', '')

    return U_moy, U_std


#    plt.xlim([0, 500])
#    plt.ylim([0, 3])

def profile_average(M, start, end, moment):
    # start and end corresponds to the bound for the average in time
    # average along the dimensions contained in dimensions for instance : ['x','y','z','t'],
    # the return function is given by the other dimensions (?)
    nx, ny, nt = M.shape()
    n = end - start

    Ux = M.Ux[:, :, start:end] / 1000
    Uy = M.Uy[:, :, start:end] / 1000

    if M.param.angle == 90:
        #   print('vertical average')
        axes = (0, 2,
                1)  # permute the time and the dimensions whose shouldn't be averaged along (here the x direction in second )
        shape = (nx, n, ny)
        z = M.x[0, :]
    if M.param.angle == 0:
        #  print('horizontal average')
        axes = (1, 2,
                0)  # permute the time and the dimensions whose shouldn't be averaged along (here the x directionm in second )
        shape = (ny, n, nx)
        z = M.y[:, 0]

    Ux = np.reshape(np.transpose(Ux, axes), (shape[0] * shape[1], shape[2]))
    Uy = np.reshape(np.transpose(Uy, axes), (shape[0] * shape[1], shape[2]))

    # RMS velocity profile along the second direction : should correspond to the z axis
    Ux_moy = np.mean(Ux, axis=0)
    Uy_moy = np.mean(Uy, axis=0)
    if moment == 1:
        return z, Ux_moy, Uy_moy
    else:
        Upx = (np.mean((Ux - Ux_moy) ** moment, axis=0)) ** (1. / moment)
        Upy = (np.mean((Uy - Uy_moy) ** moment, axis=0)) ** (1. / moment)
        return z, Upx, Upy


def profile_xy(S, x0, y0):
    # plot the profile at a given position : will be use to compare several movies
    i, j = get_index(S.m, x0, y0)
    return i, j


def histogramm(S, label='ko'):
    # complete histogramm
    nx, ny, nt = S.shape()
    U = np.sqrt(S.Ux ** 2 + S.Uy ** 2)

    V = np.arange(0, 2, 0.005)
    n, x, shape = plt.hist(np.reshape(U, nx * ny * nt), V, histtype='step')

    # add bounded values of velocity measurement
    bounded_velocity(S, True, max(n))
    plt.show(block=False)
    time.sleep(10e-3)


def velocity_fluctuation(S, nframe):
    # for each point, compute an everage variation of velocity on nframe
    # both in angle and modulus !
    nx, ny, nt = S.shape()

    dUx = np.diff(S.Ux)
    dUy = np.diff(S.Uy)

    dU = dUx ** 2 + dUy ** 2

    # cumulative sum on the increment of velocity
    dU_sum = np.cumsum(dU, axis=2)

    dU_nframe = np.sqrt((dU_sum[:, :, nframe:] - dU_sum[:, :, 0:-nframe]) / nframe)

    make_2dmovie(S, dU_nframe, 1, nt - nframe, 'Velocity_increment')


#    dU_moy=[dU_sum[:,:,i]-dU_sum[:,:,0] for i in range(1,nframe)]
#    dUr,dUtheta=Smath.cart2pol(dUx,dUy)

def velocity(M, nframe=0):
    nx, ny, nt = M.shape()

    U = (M.Ux ** 2 + M.Uy ** 2)

    if nframe == 0:
        U_nframe = U
    else:
        U_sum = np.cumsum(U, axis=2)
        U_nframe = np.sqrt((U_sum[:, :, nframe:] - U_sum[:, :, 0:-nframe]) / (nframe + 1))

    #    Dir_root = os.path.dirname(M.Sdata.fileCine)
    Dirname = 'Velocity_modulus_' + M.id.get_id()
    make_2dmovie(M, U_nframe, 1, 630, Dirname)


def time_window(S, nframe):
    # compute histogramm from each time windows
    U = np.sqrt(S.Ux ** 2 + S.Uy ** 2)
    nx, ny, nt = S.shape()
    # average every nframe :
    for i in range(nframe, nt, nframe):
        V = U[i - nframe:i]

        plt.clf()
        n, x, shape = plt.hist(np.reshape(V, nx * ny * nframe), V, histtype='step')
        bounded_velocity(S, True, max(n))

        plt.show(block=False)
        time.sleep(10e-3)
        input()


def multi_graph(function, Slist):
    for S in Slist:
        function(S)
        input()


def make_2dmovie(M, Z, start, stop, Dirname='velocity_module', vmin=-3, vmax=3):
    # Movie of the velocity modulus U
    #    U=np.sqrt(S.Ux**2+S.Uy**2)
    fignum = 5
    for i in range(start, stop):
        Z0 = Z[:, :, i] / 1000
        print(i)

        #  print('<dU>_x,y = '+str(np.median(Z0)))
        #  print('<dU^2>_x,y = '+str(np.std(Z0)))
        graphes.color_plot(M.x, M.y, np.log10(Z0), fignum=fignum, vmin=vmin, vmax=vmax)
        if i == start:
            graphes.colorbar(fignum=fignum, label='V (m/s)')
        titre = ''
        graphes.legende('x (mm)', 'y (mm)', titre)

        if i == start:
            eval(input())
        #       graphes.vfield_plot(S)

        Dir = os.path.dirname(M.Sdata.fileCine) + '/' + Dirname + '/'
        print(Dir)
        filename = Dir + 'V' + str(i)
        graphes.save_fig(fignum, filename, fileFormat='png')


def measure_anisotropy(fileDir, t0, fpms):
    fileList, n = get_fileList(fileDir, 'npz')
    I = []
    for name in fileList:
        S = np.load(name)
        # print(name)
        I0 = anisotropy(S)
        I.append(I0)

    # default fps is 1000fps in PIV parameters files
    ft = 1. / fpms
    t = np.arange(t0, t0 + n * ft, ft)
    graphes.graphloglog(t, U, label)
    graphes.legende('t (ms)', 'I (norm.)', '')
    print((str(max(t)) + '   ' + str(max(U))))
    # raw_input()


def anisotropy(S):
    Ux = np.median(S['Ux'] ** 2)
    Uy = np.median(S['Uy'] ** 2)

    Imoy = (Uy - Ux) / (Ux + Uy)
    I = (S['Uy'] ** 2 - S['Ux'] ** 2) / (S['Ux'] ** 2 + S['Uy'] ** 2)

    return Imoy


def velocity_vp(S):
    # averaged vertical profile of the Uy component
    Uy_moy = [np.mean(S['uy'][i, :]) for i in range(len(S['uy'][:, 0]))]
    y = S['y'][:, 0]

    return y, Uy_moy


def velocity_ditribution(N):
    return 0
    # N correspond to the number of slices that we will used to averaged the velocity field locally

# def main():
#   velocity(M_log[2])

# main()
