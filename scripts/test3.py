#!/usr/bin/env python

## Script to test modules
import numpy as np
import scipy
import sys
sys.path.append('/Users/stephane/Documents/git/takumi/library/')
import display.graph as graph
import argparse
import matplotlib.pyplot as plt
import string

#parser if neccesary
parser = argparse.ArgumentParser(description="This script intends to test modules under development")
parser.add_argument('-g', dest='gamma', type=float, default=1., help='gamma of output, assumes input is gamma 1, or: I -> I**(1/gamma); use 2.2 to turn linear image to "natural" for display [default: 1]')
args = parser.parse_args()


def func1(x, param):
    return param[0] * x ** param[1] + param[2]
def func2(x, param):
    return param[0] * np.exp(- x ** param[1] + param[2]) + param[3]



# test plotting modules
x = np.linspace(0, 1, 100)

xx = 10.0**np.linspace(0.0, 2.0, 20)
yy = xx**2.0
xerr=0.1*xx
yerr=5.0 + 0.75*yy


fig1, ax1 = graph.plotfunc(func1, x, param=[1,2,3], fignum=1, label='func1 param1', color='r', subplot=321, legend=False)
fig2, ax2 = graph.plotfunc(func1, x, param=[5,1,3,2], fignum=1, label='func1 param2',  subplot=321,legend=False)
fig3, ax3 = graph.plotfunc(func2, x, param=[5,1,3,2], fignum=1, label='func2', subplot=322,legend=False)
fig4, ax4 = graph.plotfunc(func2, x, param=[5,1,3,2], fignum=1,label='func2 param1', color='r', subplot=323, legend=False)
#fig5, ax5 = graph.plot(xx,2*10**3*(xx/1)**-3.0)
fig5, ax5 = graph.plot(xx, yy,fignum=1,label='plotting x,y', color='r', subplot=324, legend=False)
fig6, ax6 = graph.errorbar(xx, yy, yerr, xerr, fignum=1, label='errorbar', color='r', subplot=325, legend=False)
fig7, ax7, color_patch= graph.errorfill(xx, yy, yerr, fignum=1,  color='r', label='errorfill', subplot=326, legend=False)

graph.tologlog(ax4)
graph.tosemilogy(ax3)
graph.setaxes(ax5,0,150,0,10**4)
#graph.legend()
graph.show()
