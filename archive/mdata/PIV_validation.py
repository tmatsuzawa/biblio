# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:26:28 2015

@author: stephane
"""

# PIV_validation provide basic tools to compare several set of PIV measurements
# by individual comparison between two different software, two different frame rate, etc.

import pylab as plt
import numpy as np
# from Mdata import Mdata
import match
# npm changed below from import graphes
import turbulence.display.graphes as graphes
import browse


# from Mdata_pyPIV import Mdata
# from Mdata_PIVlab import Mdata

def bounded_velocity(S, display=False, val=1, orientation='v'):
    W = 32  # pix
    # upper bound
    Umax = W / 4 * S.fx * S.fps * S.timescale
    # lower bound
    Umin = 0.01 * S.fx * S.fps * S.timescale
    # add criterion on the gradient of velocity (!)
    # dU/dW <0.1
    if orientation == 'v':
        X1 = [Umin, Umin]
        X2 = [Umax, Umax]
        Y1 = [0, val]
        Y2 = [0, val]
    else:
        X1 = [0, val]
        X2 = [0, val]
        Y1 = [Umin, Umin]
        Y2 = [Umax, Umax]
    if display:
        plt.plot(X1, Y1, 'r--')
        plt.plot(X2, Y2, 'r--')

    return Umax, Umin


def compare_measure(M1, M2):
    indices1_t, indices2_t, nt = match.time(M1, M2)
    indices1_xy, indices2_xy, nt = match.space(M1, M2)

    for tup1 in indices1_xy:
        Ux1 = M1.Ux[tup1[0], tup1[1], indices1_t]
        Uy1 = M1.Ux[tup1[0], tup1[1], indices1_t]

        tup2 = indices2_xy[indices1_xy.index(tup1)]
        print((tup2[0]))
        print((tup2[1]))
        Ux2 = M2.Ux[tup2[0], tup2[1], indices2_t]
        Uy2 = M2.Uy[tup2[0], tup2[1], indices2_t]

        t1 = str(type(M1))
        t2 = str(type(M2))

        name1 = browse.get_string(t1, 'Mdata_', '.')
        name2 = browse.get_string(t2, 'Mdata_', '.')
        title = name1 + ' vs ' + name2

        graphes.graph(Ux1, Ux2, 1, 'ro')
        graphes.legende('$Ux $', '$Ux$', title)
        graphes.graph(Uy1, Uy2, 2, 'ro')
        graphes.legende('$Uy $', '$Uy$', title)

        plt.pause(10)


def compare_point(indices1, indices2):
    M1.Ux[indices1]


def error_map(M1, M2):
    # S1 and S2 must be the same length, and the same x and y dimensions
    # compare velocity measurement obtained with two different frame rate
    # norm and direction
    #    nx,ny,nt=S1.shape()
    indices1_t, indices2_t, nt = match.time(M1, M2)
    indices1_xy, indices2_xy, nt = match.space(M1, M2)

    Vx1 = S1.Ux[:, :, indices1]
    Vy1 = S1.Uy[:, :, indices1]

    Vx2 = S2.Ux[:, :, indices2]
    Vy2 = S2.Uy[:, :, indices2]

    U1 = np.sqrt(Vx1 ** 2 + Vy1 ** 2)
    U2 = np.sqrt(Vx2 ** 2 + Vy2 ** 2)

    dVx = (Vx1 - Vx2)  # *2/(U1+U2)
    dVy = (Vy1 - Vy2)  # *2/(U1+U2)

    delta = np.sqrt(dVx ** 2 + dVy ** 2)
    Dirname = 'Error_map_' + str(int(S1.fps)) + 'fps_vs_' + str(int(S2.fps)) + 'fps'
    make_2dmovie(S1, delta, 0, nt, Dirname)
