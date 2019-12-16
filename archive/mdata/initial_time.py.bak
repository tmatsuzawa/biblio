# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:46:42 2015

@author: stephane
"""

# Sdata,param
import turbulence.tools.read_cine as read_cine
import numpy as np
from math import sqrt

import turbulence.display.graphes as graphes


# import Sdata_measure

def get_im0(par, start=0, n=2000):
    print('set origin of time from the cine file')
    return from_cine(par.Sdata.fileCine, n + start, start)


def from_cine(fileCine, end, start, bound=10):
    frameList = read_cine.read_cine(fileCine, end, start)

    #    print("number of files :"+str(len(frameList)))
    # bound is used to count the number of pixels where the difference between two images is greater than 10

    debut = True
    lastframe = []
    I = np.zeros(0)
    count = 0
    for frame in frameList:
        #        print(str(frameList.index(frame)))
        if not debut:
            framediff = frame - lastframe
            # number of point where the difference of light is greater than bound pixels
            idex = np.sum(np.abs(framediff) > bound)
            I = np.append(I, idex)
            # print(str(count)+' : '+str(idex))
        else:
            debut = False
        lastframe = frame
        count += 1

    im0 = detect_start(I, start, fileCine)
    return im0


# def plot_detection(I,start,im0,title):


def detect_start(X, start, fileCine, epsilon=10, Dt=10):
    # start at the beginning : measure the average on the first ten images
    vstart = np.mean(X[:Dt]) + epsilon
    # endwhen the motion is the stronger
    indmax = np.argmax(X)
    vend = np.mean(X[indmax - Dt / 2:indmax + Dt / 2])
    # geometrical average
    lim = sqrt(vstart * vend)

    lbound = 2400
    lim = max([lim, lbound])
    if np.isnan(lim):
        lim = lbound
    # lim=2000


    print('Threshold : ' + str(lim))
    indices = np.where(X > lim)
    # print(np.shape(indices))
    # np.extract : find the first indices satisfying this condition
    im0 = indices[0][0] + start
    val = np.max(X)

    index = np.arange(len(X)) + start

    title = fileCine
    graphes.graph(index, X, True)
    graphes.graph([im0, im0], [0, val], False, 'r-')
    graphes.legende('t (# image)', 'grid motion (a.u.)', title)

    valid = ''
    #    valid=input('Confirm ?')
    if valid == '':
        print('ok')
    else:
        valid = input('Confirm ?')

        if not valid == '':
            print('Origin of time arbitrary set to 0')
            im0 = 0

    return im0


def from_PIV(par):
    # measure the initial time from the PIV measurement -> wrong method, depends on the PIV parameters
    # plot the
    #    i,j=measure.get_index(m,x0,y0)
    #    Sdata_measure.mean_profile(S,i,j,'h','k^',True)
    #    Sdata_measure.velocity_profile(S,[j],[i],'k^')

    return 0


def from_signal(sig):
    # need a synchronization between the different triggers
    # next step ! (if needed)
    return 0


"""
file='/Volumes/labshared3/Stephane/Experiments/2015_02_25/PIV_sv_X25mm_fps16000_H810mm_zoom_S100mm.cine'
n=1000
from_cine(file,n)

"""
