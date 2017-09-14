# -*- coding: utf-8 -*-
"""
Created on Jul 7 2017
@author: takumi
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

''' 
Script that outputs a figure of Taylor Reynolds number and Taylor scale

Currently, the script has too many dependencies. This will be cleared up eventually. 9/12/17
'''

## Compute Taylor microscale: lambda and Re_lambda
Mfluc.Ux2 = np.zeros((M.Ux.shape[0], M.Ux.shape[1], M.Ux.shape[2]))
Mfluc.Uy2 = np.zeros((M.Ux.shape[0], M.Ux.shape[1], M.Ux.shape[2]))
Mfluc.U2 = np.zeros((M.Ux.shape[0], M.Ux.shape[1], M.Ux.shape[2]))

Mfluc.dUxdx = np.zeros((M.Ux.shape[0], M.Ux.shape[1], M.Ux.shape[2]))
Mfluc.dUydy = np.zeros((M.Ux.shape[0], M.Ux.shape[1], M.Ux.shape[2]))

Mfluc.dUidxi2 = np.zeros((M.Ux.shape[0], M.Ux.shape[1], M.Ux.shape[2]))
Mfluc.dUxdx2 = np.zeros((M.Ux.shape[0], M.Ux.shape[1], M.Ux.shape[2]))
Mfluc.dUydy2 = np.zeros((M.Ux.shape[0], M.Ux.shape[1], M.Ux.shape[2]))

Mfluc.U2ave = np.zeros((M.Ux.shape[0], M.Ux.shape[1]))
Mfluc.Ux2ave = np.zeros((M.Ux.shape[0], M.Ux.shape[1]))
Mfluc.Uy2ave = np.zeros((M.Ux.shape[0], M.Ux.shape[1]))

Mfluc.dUidxi2ave = np.zeros((M.Ux.shape[0], M.Ux.shape[1]))
Mfluc.dUxdx2ave = np.zeros((M.Ux.shape[0], M.Ux.shape[1]))
Mfluc.dUydy2ave = np.zeros((M.Ux.shape[0], M.Ux.shape[1]))

Mfluc.lambdaT = np.zeros((M.Ux.shape[0], M.Ux.shape[1]))
Mfluc.lambdaTx = np.zeros((M.Ux.shape[0], M.Ux.shape[1]))
Mfluc.lambdaTy = np.zeros((M.Ux.shape[0], M.Ux.shape[1]))
Mfluc.Re_lambdaT = np.zeros((M.Ux.shape[0], M.Ux.shape[1]))
Mfluc.Re_lambdaTx = np.zeros((M.Ux.shape[0], M.Ux.shape[1]))
Mfluc.Re_lambdaTy = np.zeros((M.Ux.shape[0], M.Ux.shape[1]))
i = .0
nu = 1.004  # [mm^2/s]: kinematic viscosity of water at 20C
###
# dx=fx*(# of pixels used to compute a velocity vector in PIV... can be controlled by variable "W" in a Matlab code)
# it is easy to get dx by the following line
dx = dy = Mfluc.x[1, 2] - Mfluc.x[1, 1]  # [mm]
###
print Mfluc.Ux.shape
tmin = 6000
for t in range(tmin, M.Ux.shape[2]):
    for x in range(0, M.Ux.shape[0] - 1):
        for y in range(0, M.Ux.shape[1] - 1):

            Mfluc.Ux2[x, y, t] = Mfluc.Ux[x, y, t] * Mfluc.Ux[x, y, t]
            Mfluc.Uy2[x, y, t] = Mfluc.Uy[x, y, t] * Mfluc.Uy[x, y, t]
            Mfluc.U2[x, y, t] = (Mfluc.Ux2[x, y, t] + Mfluc.Uy2[x, y, t])
            if x == M.Ux.shape[0] - 1 or y == M.Ux.shape[1] - 1:
                continue

            Mfluc.dUxdx[x, y, t] = (Mfluc.Ux[x + 1, y, t] - Mfluc.Ux[
                x, y, t]) / dx  # du'_x/dx, convert fx[cm/px] to [mm/px]
            Mfluc.dUydy[x, y, t] = (Mfluc.Uy[x, y + 1, t] - Mfluc.Uy[
                x, y, t]) / dy  # du'_y/dy, convert fx[cm/px] to [mm/px]
            Mfluc.dUxdx2[x, y, t] = Mfluc.dUxdx[x, y, t] * Mfluc.dUxdx[x, y, t]
            Mfluc.dUydy2[x, y, t] = Mfluc.dUydy[x, y, t] * Mfluc.dUydy[x, y, t]
            Mfluc.dUidxi2[x, y, t] = Mfluc.dUxdx[x, y, t] * Mfluc.dUxdx[x, y, t] + Mfluc.dUydy[x, y, t] * Mfluc.dUydy[
                x, y, t]

##Time average of u^2 and (du/dx)^2
## np.nanmean does not output proper mean when inf is contained in the arrays
# Mfluc.U2ave = np.nanmean(Mfluc.U2[...,indices],axis=2)
# Mfluc.dUidxi2ave = np.nanmean(Mfluc.dUidxi2[...,indices],axis=2)

counter_1 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
counter_2 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
counter_3 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
counter_4 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
counter_5 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
counter_6 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))

for t in range(tmin, Mfluc.Ux.shape[2]):
    for x in range(0, Mfluc.Ux.shape[0]):
        for y in range(0, Mfluc.Ux.shape[1]):
            #             if np.isnan(Mfluc.U2[x,y,t])==False and np.isinf(Mfluc.U2[x,y,t])==False:
            #                 Mfluc.U2ave[x,y] += Mfluc.U2[x,y,t]
            #                 counter_1[x,y]+=1
            if np.isnan(Mfluc.Ux2[x, y, t]) == False and np.isinf(Mfluc.Ux2[x, y, t]) == False:
                Mfluc.Ux2ave[x, y] += Mfluc.Ux2[x, y, t]
                counter_2[x, y] += 1
            if np.isnan(Mfluc.Uy2[x, y, t]) == False and np.isinf(Mfluc.Uy2[x, y, t]) == False:
                Mfluc.Uy2ave[x, y] += Mfluc.Uy2[x, y, t]
                counter_3[x, y] += 1

            # if np.isnan(Mfluc.dUidxi2[x,y,t])==False and np.isinf(Mfluc.dUidxi2[x,y,t])==False:
            #                 counter_4[x,y]+=1
            #                 Mfluc.dUidxi2ave[x,y] += Mfluc.dUidxi2[x,y,t]
            if np.isnan(Mfluc.dUxdx2[x, y, t]) == False and np.isinf(Mfluc.dUxdx2[x, y, t]) == False:
                Mfluc.dUxdx2ave[x, y] += Mfluc.dUxdx2[x, y, t]
                counter_5[x, y] += 1
            if np.isnan(Mfluc.dUydy2[x, y, t]) == False and np.isinf(Mfluc.dUydy2[x, y, t]) == False:
                Mfluc.dUydy2ave[x, y] += Mfluc.dUydy2[x, y, t]
                counter_6[x, y] += 1

print ('Calculating the mean U2 and dUidxi2...')

for x in range(0, Mfluc.Ux.shape[0]):
    for y in range(0, Mfluc.Ux.shape[1]):
        #         if counter_1[x,y]==0:
        #             Mfluc.U2ave[x,y] = 0
        #         else:
        #             Mfluc.U2ave[x,y] = Mfluc.U2ave[x,y]/counter_1[x,y]
        if counter_2[x, y] == 0:
            Mfluc.Ux2ave[x, y] = 0
        else:
            Mfluc.Ux2ave[x, y] = Mfluc.Ux2ave[x, y] / counter_2[x, y]
        if counter_3[x, y] == 0:
            Mfluc.Uy2ave[x, y] = 0
        else:
            Mfluc.Uy2ave[x, y] = Mfluc.Uy2ave[x, y] / counter_3[x, y]

        # if counter_4[x,y]==0:
        #             Mfluc.dUidxi2ave[x,y] = 0
        #         else:
        #             Mfluc.dUidxi2ave[x,y] = Mfluc.dUidxi2ave[x,y]/counter_4[x,y]
        if counter_5[x, y] == 0:
            Mfluc.dUxdx2ave[x, y] = 0
        else:
            Mfluc.dUxdx2ave[x, y] = Mfluc.dUxdx2ave[x, y] / counter_5[x, y]
        if counter_6[x, y] == 0:
            Mfluc.dUydy2ave[x, y] = 0
        else:
            Mfluc.dUydy2ave[x, y] = Mfluc.dUydy2ave[x, y] / counter_6[x, y]

Mfluc.U2ave[...] = (Mfluc.Ux2ave[...] + Mfluc.Uy2ave[...]) / 2
Mfluc.dUidxi2ave[...] = (Mfluc.dUxdx2ave[...] + Mfluc.dUydy2ave[...]) / 2
# print Mfluc.U2ave
# print Mfluc.dUidxi2ave

# Calculate Taylor microscale: lambdaT
for x in range(0, M.Ux.shape[0] - 1):
    for y in range(0, M.Ux.shape[1] - 1):
        if Mfluc.dUidxi2ave[x, y] == 0:
            Mfluc.lambdaT[x, y] = 0
        else:
            Mfluc.lambdaT[x, y] = np.sqrt(Mfluc.U2ave[x, y] / Mfluc.dUidxi2ave[x, y])
            Mfluc.Re_lambdaT[x, y] = Mfluc.lambdaT[x, y] * np.sqrt(Mfluc.U2ave[x, y]) / nu
        if Mfluc.dUxdx2ave[x, y] == 0:
            Mfluc.lambdaTx[x, y] = 0
        else:
            Mfluc.lambdaTx[x, y] = np.sqrt(Mfluc.Ux2ave[x, y] / Mfluc.dUxdx2ave[x, y])
            Mfluc.Re_lambdaTx[x, y] = Mfluc.lambdaTx[x, y] * np.sqrt(Mfluc.Ux2ave[x, y]) / nu
        if Mfluc.dUydy2ave[x, y] == 0:
            Mfluc.lambdaTy[x, y] = 0
        else:
            Mfluc.lambdaTy[x, y] = np.sqrt(Mfluc.Uy2ave[x, y] / Mfluc.dUydy2ave[x, y])
            Mfluc.Re_lambdaTy[x, y] = Mfluc.lambdaTy[x, y] * np.sqrt(Mfluc.Uy2ave[x, y]) / nu

            ## Plot Taylor miroscale
# graphes.color_plot(Mfluc.x,Mfluc.y,Mfluc.lambdaT,fignum=j+1,vmin=0,vmax=2)
# graphes.colorbar()
# plt.title('lambda')
# plt.xlabel('X (mm)')
# plt.ylabel('Y (mm)')

graphes.color_plot(Mfluc.x, Mfluc.y, Mfluc.Re_lambdaT, fignum=1, vmin=0, vmax=100)
graphes.colorbar()
plt.title('$Re_\lambda$')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')