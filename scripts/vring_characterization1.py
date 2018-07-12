#!/usr/bin/env python

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


# fig1, ax1 = graph.plotfunc(func1, x, param=[1,2,3], fignum=1, label='func1 param1', color='r', subplot=321, legend=False)
# fig2, ax2 = graph.plotfunc(func1, x, param=[5,1,3,2], fignum=1, label='func1 param2',  subplot=321,legend=False)
# fig3, ax3 = graph.plotfunc(func2, x, param=[5,1,3,2], fignum=1, label='func2', subplot=322,legend=False)
# fig4, ax4 = graph.plotfunc(func2, x, param=[5,1,3,2], fignum=1,label='func2 param1', color='r', subplot=323, legend=False)
# fig5, ax5 = graph.plot(xx, yy,fignum=1,label='plotting x,y', color='r', subplot=324, legend=False)
# fig6, ax6 = graph.errorbar(xx, yy, yerr, xerr, fignum=1, label='errorbar', color='r', subplot=325, legend=False)
# fig7, ax7 = graph.errorfill(xx, yy, yerr, fignum=1,  color='r', label='errorfill', subplot=326, legend=False)
# graph.tologlog(ax4)
# graph.tosemilogy(ax3)
# graph.setaxis(ax5,0,150,0,10**4)
# graph.show()

err_corr = True

vp8_tilde = [50.80023391, 93.19817432, 149.1035182, 206.8381824, 267.1139102, 309.110367, 355.1807229, 404.4628669,
             482.208083]
vring8 = [289.8332719, 607.8422003, 759.3658974, 871.2635114, 932.627611, 1002.605918, 1045.273717, 1109.0001,
          1158.094993]
vring8_err = [41.35606144, 14.393655, 72.02204278, 19.1293918, 108.1654841, 97.82713323, 0, 0, 0]  # stdv
vp8_tilde_tr = [50.80023391, 93.19817432, 149.1035182, 206.8381824, 309.110367, 404.4628669, 482.208083]
vring8_tr = [303.5550896, 522.2753931, 626.9929855, 814.9802547, 1102.018298, 1298.189639, 1405.386037]
vring8_tr_err = [0, 0, 0, 0, 0, 0, 0]  # stdv

vp5_tilde = [52.20796835, 100.3012048, 152.6983848, 183.5577605, 220.8835341, 252.3748842, 275.6638063, 294.4024903,
             323.8975319]
vring5 = [310.6873169, 520.3487062, 665.9826391, 744.4217064, 787.7771141, 843.4672612, 877.2749729, 919.8596988,
          953.3898419]
vring5_err = [0, 2.899709008, 19.14941055, 26.09833404, 5.103495984, 0, 0, 6.878383116, 25.67266118]  # stdv
vring5_tr = [300.892778, 534.3510144, 650.4810802, 797.8848173, 895.1201167, 971.9361107, 1029.201459, 1069.853039,
             1097.048098]
vring5_tr_err = [0, 0, 0.45555622, 0, 0, 0, 0, 4.94627501, 8.929268574]

vp2_tilde = [51.80611675, 102.2695433, 187.4315444, 229.3476178, 261.3023931, 301.5976951]
vring2 = [177.6735118, 272.1010729, 297.907515,  310.4320921, 310.4320921, 324.245004]
vring2_err = [0, 0, 18.90155006, 0, 0,  8.75499598]  # stdv
vring2_tr = [190.2898977, 218.2932695, 227.6568166,  262.803588, 263.1226858, 272.7384014]
vring2_tr_err = [13.99746773, 13.56977493, 1.497502466, 10.06250147, 1.910570935,  5.41946662]  # stdv

if err_corr:
    for i in range(len(vring8_err)):
        if vring8_err[i] == 0:
            vring8_err[i] = vring8[i] * 0.04
    for i in range(len(vring8_tr_err)):
        if vring8_tr_err[i] == 0:
            vring8_tr_err[i] = vring8_tr[i] * 0.04

    for i in range(len(vring5_err)):
        if vring5_err[i] == 0:
            vring5_err[i] = vring5[i] * 0.02
    for i in range(len(vring5_tr_err)):
        if vring5_tr_err[i] == 0:
            vring5_tr_err[i] = vring5_tr[i] * 0.02

    for i in range(len(vring2_err)):
        if vring2_err[i] == 0:
            vring2_err[i] = vring2[i] * 0.05
    for i in range(len(vring2_tr_err)):
        if vring2_tr_err[i] == 0:
            vring2_tr_err[i] = vring2_tr[i] * 0.05

#graph.test(fignum=10, dpi=10,figsize=(10, 100))
figsize=(8,36)
xmin, xmax, ymin, ymax = 0., 500., 0., 1500.
fig1, ax1, color_patch1 = graph.errorfill(vp8_tilde_tr, vring8_tr, yerr=vring8_tr_err, marker='None', linewidth=2, linestyle='None',
                 label='L/D=2.11, top', markersize=10,  subplot=311, figsize=figsize)  # span=8mm
fig2, ax2, color_patch2 = graph.errorfill(vp8_tilde, vring8, yerr=vring8_err, marker='None', linewidth=2, linestyle='None',
                label='L/D=2.11, bottom', markersize=10, subplot=311,figsize=figsize)  # span=8mm
graph.setaxes(ax1, xmin, xmax, ymin, ymax)
graph.labelaxes(xlabel='$\overline{v_p^2}$ / ${\overline{v_p}}$ [mm/s]',ylabel='vortex ring speed [mm/s]', fontsize=10)
plt.legend(handles=[color_patch1, color_patch2])

fig3, ax3, color_patch3 = graph.errorfill(vp5_tilde, vring5_tr, yerr=vring5_tr_err, marker='None', linewidth=2, linestyle='None',
                 label='L/D=1.32, top', markersize=10,  subplot=312,figsize=figsize)  # span=5mm
fig4, ax4, color_patch4 = graph.errorfill(vp5_tilde, vring5, yerr=vring5_err, marker='None', linewidth=2, linestyle='None',
                label='L/D=1.32, bottom', markersize=10, subplot=312,figsize=figsize)  # span=5mm
graph.setaxes(ax3, xmin, xmax, ymin, ymax)
graph.labelaxes(xlabel='$\overline{v_p^2}$ / ${\overline{v_p}}$ [mm/s]',ylabel='vortex ring speed [mm/s]',fontsize=10)
plt.legend(handles=[color_patch3, color_patch4])

fig5, ax5, color_patch5 = graph.errorfill(vp2_tilde, vring2_tr, yerr=vring2_tr_err, marker='None', linewidth=2, linestyle='None',
                 label='L/D=0.53, top', markersize=10,  subplot=313,figsize=figsize)  # span=2mm
fig6, ax6, color_patch6 = graph.errorfill(vp2_tilde, vring2, yerr=vring2_err, marker='None', linewidth=2, linestyle='None',
                label='L/D=0.53, bottom', markersize=10, subplot=313,figsize=figsize)  # span=2mm
graph.setaxes(ax5, xmin, xmax, ymin, ymax)
graph.labelaxes(xlabel='$\overline{v_p^2}$ / ${\overline{v_p}}$ [mm/s]',ylabel='vortex ring speed [mm/s]',fontsize=10)
plt.legend(handles=[color_patch5, color_patch6])

graph.show()

savedir = "/Volumes/labshared3-1/takumi/good_data/vortex_ring_characterization/figures/"
filename = "vtop_vbottom"
filepath = savedir + filename
#graph.save(filepath)

#
#
# xmin, xmax, ymin, ymax = 0., 500., 0., 1500.
#
# fig1, ax1, color_patch1 = graph.errorfill(vp8_tilde_tr, vring8_tr, yerr=vring8_tr_err, marker='None', linewidth=2, linestyle='None',
#                  label='L/D=2.11, top', markersize=10,  subplot=311, figsize=figsize,fignum=2)  # span=8mm
# fig2, ax2, color_patch2 = graph.errorfill(vp8_tilde, vring8, yerr=vring8_err, marker='None', linewidth=2, linestyle='None',
# #                 label='L/D=2.11, bottom', markersize=10, subplot=311,figsize=figsize,fignum=2)  # span=8mm
# graph.setaxes(ax1, xmin, xmax, ymin, ymax)
# graph.labelaxes(xlabel='$\overline{v_p^2}$ / ${\overline{v_p}}$ [mm/s]',ylabel='vortex ring speed [mm/s]', fontsize=10)
# plt.legend(handles=[color_patch1, color_patch2])
#
# fig3, ax3, color_patch3 = graph.errorfill(vp5_tilde, vring5_tr, yerr=vring5_tr_err, marker='None', linewidth=2, linestyle='None',
#                  label='L/D=1.32, top', markersize=10,  subplot=312,figsize=figsize,fignum=2)  # span=5mm
# fig4, ax4, color_patch4 = graph.errorfill(vp5_tilde, vring5, yerr=vring5_err, marker='None', linewidth=2, linestyle='None',
#                 label='L/D=1.32, bottom', markersize=10, subplot=312,figsize=figsize,fignum=2)  # span=5mm
# graph.setaxes(ax3, xmin, xmax, ymin, ymax)
# graph.labelaxes(xlabel='$\overline{v_p^2}$ / ${\overline{v_p}}$ [mm/s]',ylabel='vortex ring speed [mm/s]',fontsize=10)
# plt.legend(handles=[color_patch3, color_patch4])
#
# fig5, ax5, color_patch5 = graph.errorfill(vp2_tilde, vring2_tr, yerr=vring2_tr_err, marker='None', linewidth=2, linestyle='None',
#                  label='L/D=0.53, top', markersize=10,  subplot=313,figsize=figsize,fignum=2)  # span=2mm
# fig6, ax6, color_patch6 = graph.errorfill(vp2_tilde, vring2, yerr=vring2_err, marker='None', linewidth=2, linestyle='None',
#                 label='L/D=0.53, bottom', markersize=10, subplot=313,figsize=figsize,fignum=2)  # span=2mm
# graph.setaxes(ax5, xmin, xmax, ymin, ymax)
# graph.labelaxes(xlabel='$\overline{v_p^2}$ / ${\overline{v_p}}$ [mm/s]',ylabel='vortex ring speed [mm/s]',fontsize=10)
# plt.legend(handles=[color_patch5, color_patch6])
# graph.show()
#
#
# savedir = "/Volumes/labshared3-1/takumi/good_data/vortex_ring_characterization/figures/"
# filename = "vtop_vbottom2"
# filepath = savedir + filename
# graph.save(filepath)

#
# xmin, xmax, ymin, ymax = 0., 500., 0., 1500.
# fig7, ax7, color_patch7 = graph.errorfill(vp8_tilde_tr, vring8_tr, yerr=vring8_tr_err, marker='None', linewidth=2, linestyle='None',
#                  label='L/D=2.11, top', markersize=10,  subplot=324, figsize=figsize)  # span=8mm
# fig8, ax8, color_patch8 = graph.errorfill(vp8_tilde, vring8, yerr=vring8_err, marker='None', linewidth=2, linestyle='None',
#                 label='L/D=2.11, bottom', markersize=10, subplot=324,figsize=figsize)  # span=8mm
# graph.setaxes(ax1, xmin, xmax, ymin, ymax)
# graph.labelaxes(xlabel='$\overline{v_p^2}$ / ${\overline{v_p}}$ [mm/s]',ylabel='vortex ring speed [mm/s]', fontsize=10)
# plt.legend(handles=[color_patch7, color_patch8])
#
# fig9, ax9, color_patch9 = graph.errorfill(vp5_tilde, vring5_tr, yerr=vring5_tr_err, marker='None', linewidth=2, linestyle='None',
#                  label='L/D=1.32, top', markersize=10,  subplot=325,figsize=figsize)  # span=5mm
# fig10, ax19, color_patch10 = graph.errorfill(vp5_tilde, vring5, yerr=vring5_err, marker='None', linewidth=2, linestyle='None',
#                 label='L/D=1.32, bottom', markersize=10, subplot=325,figsize=figsize)  # span=5mm
# graph.setaxes(ax3, xmin, xmax, ymin, ymax)
# graph.labelaxes(xlabel='$\overline{v_p^2}$ / ${\overline{v_p}}$ [mm/s]',ylabel='vortex ring speed [mm/s]',fontsize=10)
# plt.legend(handles=[color_patch9, color_patch10])
#
# fig11, ax11, color_patch11 = graph.errorfill(vp2_tilde, vring2_tr, yerr=vring2_tr_err, marker='None', linewidth=2, linestyle='None',
#                  label='L/D=0.53, top', markersize=10,  subplot=326,figsize=figsize)  # span=2mm
# fig12, ax12, color_patch12 = graph.errorfill(vp2_tilde, vring2, yerr=vring2_err, marker='None', linewidth=2, linestyle='None',
#                 label='L/D=0.53, bottom', markersize=10, subplot=326,figsize=figsize)  # span=2mm
# graph.setaxes(ax5, xmin, xmax, ymin, ymax)
# graph.labelaxes(xlabel='$\overline{v_p^2}$ / ${\overline{v_p}}$ [mm/s]',ylabel='vortex ring speed [mm/s]',fontsize=10)
# plt.legend(handles=[color_patch11, color_patch12])



#
#
# savedir = "/Volumes/labshared3-1/takumi/good_data/vortex_ring_characterization/figures/"
# filename = "vtop_vbottom3"
# filepath = savedir + filename
# graph.save(filepath)






            # import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.arange(6)
# y = np.arange(5)
# z = x * y[:, np.newaxis]

# from mpl_toolkits.mplot3d import axes3d
# import matplotlib.pyplot as plt
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # load some test data for demonstration and plot a wireframe
# X, Y, Z = axes3d.get_test_data(0.1)
# ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
#
# # rotate the axes and update
# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)