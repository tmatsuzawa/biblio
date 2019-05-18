# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:14:46 2015

@author: stephane
"""

import turbulence.mdata.Sdata_measure as Sdata_measures
import numpy as np
import turbulence.display.graphes as graphes
import pylab as plt


def map_t0(M, display=False):
    nx, ny, nt = M.shape()

    #    t0=np.zeros((ny,nx))
    tmax = np.zeros((ny, nx))
    Vmax = np.zeros((ny, nx))

    for j in range(nx):
        print(j / nx * 100)
        for i in range(ny):
            if display:
                plt.clf()
            t, Ut = Sdata_measure.velocity_profile(M, [i], [j], display, 1000, 6000)
            tmax[i, j], Vmax[i, j] = detect_max(t, Ut[-1], display)
            # start(t,Ut[-1],0.2,display)
    return tmax, Vmax


def detect_max(t, X, display=False):
    vmax = np.max(X)
    indmax = np.argmax(X)
    tmax = t[indmax]

    if display:
        graphes.graph([tmax, tmax], [0, vmax], False, 'r-')
        graphes.legende('t (ms)', 'U (m/s)', '')

    return tmax, vmax


def detect_start(t, X, threshold, display=False):
    # initial velocity : average on Dt points
    Dt = 10
    vstart = np.mean(X[:Dt])
    # endwhen the motion is the stronger
    indmax = np.argmax(X)
    vmax = np.max(X)

    vend = np.mean(X[indmax - Dt / 2:indmax + Dt / 2])
    # geometrical average
    lim = np.sqrt(vstart * vend)

    lbound = 0.02  # in mn/s
    if np.isnan(lim):
        lim = lbound
        # lim=2000
        # print(lim)

    indices = np.where(X > lim)
    # print(np.shape(indices))
    # np.extract : find the first indices satisfying this condition
    if not (indices[0].size == 0):
        ind0 = indices[0][0]
    else:
        ind0 = np.argmax(t)
    t0 = t[ind0]

    if display:
        graphes.graph([t0, t0], [0, vmax], False, 'r-')
        graphes.legende('t (ms)', 'U (m/s)', '')

    return t0, vmax


def local_transport(M, i, j):
    # compare the velocity profile in two locations, and compute the temporal correlation :
    # deduce an advection coefficient, and a diffusion coefficient
    xlines = [39]  # np.arange(17,23)
    ylines = np.arange(44, 48)  # [27]
    t, Uxt, Uyt = Sdata_measure.velocity_profile_xy(M, xlines, ylines, True)


"""   
    title=fileCine
    graphes.graph(index,X,True)
    

    plt.show(False)    
    
    valid=''
#    valid=input('Confirm ?')
    if valid=='':
        print('ok')
    else:
        plt.show(True)
        valid=input('Confirm ?')
        
        if not valid=='':
            print('Origin of time arbitrary set to 0')
            im0=0
        
    return im0
    
    """
