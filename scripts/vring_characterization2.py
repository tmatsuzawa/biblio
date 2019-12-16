#!/usr/bin/env python

## Script to test modules
import numpy as np
import scipy
import sys
sys.path.append('/Users/stephane/Documents/git/takumi/library/')
import display.graph as graph
import argparse
import matplotlib.pyplot as plt
import tools.handle_data as dhandle
import string

from scipy.optimize import curve_fit

#parser if neccesary
parser = argparse.ArgumentParser(description="This script intends to test modules under development")
parser.add_argument('-g', dest='gamma', type=float, default=1., help='gamma of output, assumes input is gamma 1, or: I -> I**(1/gamma); use 2.2 to turn linear image to "natural" for display [default: 1]')
args = parser.parse_args()


# def func1(x, param):
#     return param[0] * x ** param[1] + param[2]
# def func2(x, param):
#     return param[0] * np.exp(- x ** param[1] + param[2]) + param[3]

#Fit function
def func1(x, a, b,c):
    return a*x**b+c

def func2(x, a, b):
    return a*x+b

def func3(x, a, b):
    return a*x*(np.log(x)+b)

def func4(x, a, b):
    return a*x**(0.333)+b

def func5(x, a, b):
    return a*x**(b)


load = True
# Load data
if load:
    err_corr = True

    dataDir = '/Volumes/labshared3-1/takumi/201709_vortex_ring_characterization/eight_vortex_rings/vring/'
    dataFile = '20170918_data_tr_fv_fps4020_left_macro105mm_D20mm_span5mm_freq1Hz_vVaried_eight_vortex_rings.txt'
    dataPath = dataDir + dataFile
    key, data5mmEight, counter5mmEight = dhandle.generate_data_dct(dataPath, separation='\t')
    # print counter5mmEight
    # print key
    veff5_tr_eight = data5mmEight["var6"]
    veff5_tl_eight = data5mmEight["var6"]
    veff5_br_eight = data5mmEight["var6"]
    veff5_bl_eight = data5mmEight["var6"]
    vring5_tr_eight = data5mmEight["var9"]
    vring5_tl_eight = data5mmEight["var11"]
    vring5_br_eight = data5mmEight["var13"]
    vring5_bl_eight = data5mmEight["var15"]

    err=0.04
    # veff5_tr_eighterr = data5mmEight["var6"]*err
    # veff5_tl_eighterr = data5mmEight["var6"]*err
    # veff5_br_eighterr = data5mmEight["var6"]*err
    # veff5_bl_eighterr = data5mmEight["var6"]*err
    # vring5_tr_eighterr = data5mmEight["var9"]*err
    # vring5_tl_eighterr = data5mmEight["var11"]*err
    # vring5_br_eighterr = data5mmEight["var13"]*err
    # vring5_bl_eighterr = data5mmEight["var15"]*err
    # ------------------------------------------------------------------------------------
    # Load data for Span = 8mm
    dataDir = '/Volumes/labshared3-1/takumi/201709_vortex_ring_characterization/eight_vortex_rings/vring/'
    dataFile = '20170918_data_tr_fv_fps4020_left_macro105mm_D20mm_span8mm_freq1Hz_vVaried_eight_vortex_rings.txt'
    dataPath = dataDir + dataFile
    key, data8mmEight, counter8mmEight = dhandle.generate_data_dct(dataPath, separation='\t')
    # print counter5mmEight
    print(key)

    veff8_tr_eight = data8mmEight["var6"]
    veff8_tl_eight = data8mmEight["var6"]
    veff8_br_eight = data8mmEight["var6"]
    veff8_bl_eight = data8mmEight["var6"]
    vring8_tr_eight = data8mmEight["var9"]
    vring8_tl_eight = data8mmEight["var11"]
    vring8_br_eight = data8mmEight["var13"]
    vring8_bl_eight = data8mmEight["var15"]

    # veff8_tr_eighterr = data8mmEight["var6"]*err
    # veff8_tl_eighterr = data8mmEight["var6"]*err
    # veff8_br_eighterr = data8mmEight["var6"]*err
    # veff8_bl_eighterr = data8mmEight["var6"]*err
    # vring8_tr_eighterr = data8mmEight["var9"]*err
    # vring8_tl_eighterr = data8mmEight["var11"]*err
    # vring8_br_eighterr = data8mmEight["var13"]*err
    # vring8_bl_eighterr = data8mmEight["var15"]*err

    # ------------------------------------------------------------------------------------

    ##Mask values below a certain threshold (Span = 5mm)-----------------------------------------------
    threshold = 0
    # Use numpy array for masking
    veff5_tr_eight_masked = np.ma.array(veff5_tr_eight)
    veff5_tl_eight_masked = np.ma.array(veff5_tl_eight)
    veff5_br_eight_masked = np.ma.array(veff5_br_eight)
    veff5_bl_eight_masked = np.ma.array(veff5_bl_eight)
    vring5_tr_eight_masked = np.ma.array(vring5_tr_eight)
    vring5_tl_eight_masked = np.ma.array(vring5_tl_eight)
    vring5_br_eight_masked = np.ma.array(vring5_br_eight)
    vring5_bl_eight_masked = np.ma.array(vring5_bl_eight)

    # Mask
    veff5_tr_eight_masked = np.ma.masked_where(vring5_tr_eight_masked < threshold, veff5_tr_eight_masked)
    veff5_tl_eight_masked = np.ma.masked_where(vring5_tl_eight_masked < threshold, veff5_tl_eight_masked)
    veff5_br_eight_masked = np.ma.masked_where(vring5_br_eight_masked < threshold, veff5_br_eight_masked)
    veff5_bl_eight_masked = np.ma.masked_where(vring5_bl_eight_masked < threshold, veff5_bl_eight_masked)
    vring5_tr_eight_masked = np.ma.masked_where(vring5_tr_eight_masked < threshold, vring5_tr_eight_masked)
    vring5_tl_eight_masked = np.ma.masked_where(vring5_tl_eight_masked < threshold, vring5_tl_eight_masked)
    vring5_br_eight_masked = np.ma.masked_where(vring5_br_eight_masked < threshold, vring5_br_eight_masked)
    vring5_bl_eight_masked = np.ma.masked_where(vring5_bl_eight_masked < threshold, vring5_bl_eight_masked)

    # Err
    veff5_tr_eight_err = np.ma.masked_where(vring5_tr_eight_masked < threshold, veff5_tr_eight_masked) * err
    veff5_tl_eight_err = np.ma.masked_where(vring5_tl_eight_masked < threshold, veff5_tl_eight_masked)  * err
    veff5_br_eight_err  = np.ma.masked_where(vring5_br_eight_masked < threshold, veff5_br_eight_masked) * err
    veff5_bl_eight_err  = np.ma.masked_where(vring5_bl_eight_masked < threshold, veff5_bl_eight_masked) * err
    vring5_tr_eight_err  = np.ma.masked_where(vring5_tr_eight_masked < threshold, vring5_tr_eight_masked) * err
    vring5_tl_eight_err  = np.ma.masked_where(vring5_tl_eight_masked < threshold, vring5_tl_eight_masked) * err
    vring5_br_eight_err  = np.ma.masked_where(vring5_br_eight_masked < threshold, vring5_br_eight_masked) * err
    vring5_bl_eight_err  = np.ma.masked_where(vring5_bl_eight_masked < threshold, vring5_bl_eight_masked) * err

    ##Mask values below a certain threshold (Span = 8mm)-----------------------------------------------
    # Use numpy array for masking
    veff8_tr_eight_masked = np.ma.array(veff8_tr_eight)
    veff8_tl_eight_masked = np.ma.array(veff8_tl_eight)
    veff8_br_eight_masked = np.ma.array(veff8_br_eight)
    veff8_bl_eight_masked = np.ma.array(veff8_bl_eight)
    vring8_tr_eight_masked = np.ma.array(vring8_tr_eight)
    vring8_tl_eight_masked = np.ma.array(vring8_tl_eight)
    vring8_br_eight_masked = np.ma.array(vring8_br_eight)
    vring8_bl_eight_masked = np.ma.array(vring8_bl_eight)
    # Mask
    veff8_tr_eight_masked = np.ma.masked_where(vring8_tr_eight_masked < threshold, veff8_tr_eight_masked)
    veff8_tl_eight_masked = np.ma.masked_where(vring8_tl_eight_masked < threshold, veff8_tl_eight_masked)
    veff8_br_eight_masked = np.ma.masked_where(vring8_br_eight_masked < threshold, veff8_br_eight_masked)
    veff8_bl_eight_masked = np.ma.masked_where(vring8_bl_eight_masked < threshold, veff8_bl_eight_masked)
    vring8_tr_eight_masked = np.ma.masked_where(vring8_tr_eight_masked < threshold, vring8_tr_eight_masked)
    vring8_tl_eight_masked = np.ma.masked_where(vring8_tl_eight_masked < threshold, vring8_tl_eight_masked)
    vring8_br_eight_masked = np.ma.masked_where(vring8_br_eight_masked < threshold, vring8_br_eight_masked)
    vring8_bl_eight_masked = np.ma.masked_where(vring8_bl_eight_masked < threshold, vring8_bl_eight_masked)

    # Err
    veff8_tr_eight_err = np.ma.masked_where(vring8_tr_eight_masked < threshold, veff8_tr_eight_masked) * err
    veff8_tl_eight_err = np.ma.masked_where(vring8_tl_eight_masked < threshold, veff8_tl_eight_masked) * err
    veff8_br_eight_err = np.ma.masked_where(vring8_br_eight_masked < threshold, veff8_br_eight_masked) * err
    veff8_bl_eight_err = np.ma.masked_where(vring8_bl_eight_masked < threshold, veff8_bl_eight_masked) * err
    vring8_tr_eight_err = np.ma.masked_where(vring8_tr_eight_masked < threshold, vring8_tr_eight_masked) * err
    vring8_tl_eight_err = np.ma.masked_where(vring8_tl_eight_masked < threshold, vring8_tl_eight_masked) * err
    vring8_br_eight_err = np.ma.masked_where(vring8_br_eight_masked < threshold, vring8_br_eight_masked) * err
    vring8_bl_eight_err = np.ma.masked_where(vring8_bl_eight_masked < threshold, vring8_bl_eight_masked) * err
    # ------------------------------------------------------------------------------------

    # if err_corr:
    #     for i in range(len(veff8_tr_eight_err)):
    #         if vring8_err[i] == 0:
    #             vring8_err[i] = vring8[i] * 0.04
    #     for i in range(len(vring8_tr_err)):
    #         if vring8_tr_err[i] == 0:
    #             vring8_tr_err[i] = vring8_tr[i] * 0.04
    #
    #     for i in range(len(vring5_err)):
    #         if vring5_err[i] == 0:
    #             vring5_err[i] = vring5[i] * 0.02
    #     for i in range(len(vring5_tr_err)):
    #         if vring5_tr_err[i] == 0:
    #             vring5_tr_err[i] = vring5_tr[i] * 0.02
    #
    #     for i in range(len(vring2_err)):
    #         if vring2_err[i] == 0:
    #             vring2_err[i] = vring2[i] * 0.05
    #     for i in range(len(vring2_tr_err)):
    #         if vring2_tr_err[i] == 0:
    #             vring2_tr_err[i] = vring2_tr[i] * 0.05




##Plot Vring (Top Right and Top Left) for Span=5mm and 5mm-----------------------------------------------
#fig1=plt.figure(figsize=(12,5))

#plot=fig1.add_subplot(111)
# plot.tick_params(axis='both', which='major', labelsize=16)
# plot.tick_params(axis='both', which='minor', labelsize=12)

figsize=(10,12)
xmin, xmax, ymin, ymax = 0., 500., 0., 1500.
fig1, ax1, color_patch1 = graph.errorfill(veff8_tr_eight_masked,vring8_tr_eight_masked, vring8_tr_eight_err, marker='None', linewidth=2, linestyle='None',
                 label='L/D=2.11, top right', markersize=10,  subplot=211, figsize=figsize)  # span=8mm
fig2, ax2, color_patch2 = graph.errorfill(veff8_tl_eight_masked,vring8_tl_eight_masked, vring8_tl_eight_err, marker='None', linewidth=2, linestyle='None',
                 label='L/D=2.11, top left', markersize=10,  subplot=211, figsize=figsize)  # span=8mm
fig3, ax3, color_patch3 = graph.errorfill(veff8_br_eight_masked,vring8_br_eight_masked, vring8_br_eight_err, marker='None', linewidth=2, linestyle='None',
                 label='L/D=2.11, bottom right', markersize=10,  subplot=211, figsize=figsize)  # span=8mm
fig4, ax4, color_patch4 = graph.errorfill(veff8_bl_eight_masked,vring8_bl_eight_masked, vring8_bl_eight_err, marker='None', linewidth=2, linestyle='None',
                 label='L/D=2.11, bottom left', markersize=10,  subplot=211, figsize=figsize)  # span=8mm


graph.labelaxes(xlabel='$\overline{v_p^2}$ / ${\overline{v_p}}$ [mm/s]',ylabel='vortex ring speed [mm/s]', fontsize=10)
plt.legend(handles=[color_patch1, color_patch2, color_patch3, color_patch4])
graph.setaxes(ax1, xmin, xmax, ymin, ymax)



#
# #Add fit curve: a(xln(x)+b)-----------------------------------------
# popt, pcov = curve_fit(func3, veff8_tr_eight_masked,vring8_tr_eight_masked, bounds=([0,-100], [1. , 100]))
# x=np.arange(1, 500,5)
# y=func3(x,*popt)
#
# fit_param=list()
# for item in popt:
#     fit_param.append(str(round(item,4)))
# fit_eq='$y=a x (ln(x)+b)$: $a$='+ fit_param[0] +', $b$=' + fit_param[1]
# print 'fit parameters: a*veff*ln(veff)+b:'  #approx
# print popt
# plt.plot(x,y,'r--',label='fit'+fit_eq)
# graph.addtext(fig=fig1, subplot=211, text=fit_eq, x=280,y=300,fontsize=10,color='r')
#
#
# #------------------------------------------------------------------------------------
# #Add fit curve: a(xln(x)+b)-----------------------------------------
# veff8_tr_eight_masked_fit = veff8_tr_eight_masked[3:len(veff8_tr_eight_masked)]
# vring8_tr_eight_masked_fit = vring8_tr_eight_masked[3:len(vring8_tr_eight_masked)]
#
# popt, pcov = curve_fit(func2, veff8_tr_eight_masked_fit, vring8_tr_eight_masked_fit)
# x=np.arange(1, 500,5)
# y=func2(x,*popt)
#
# fit_param=list()
# for item in popt:
#     fit_param.append(str(round(item,4)))
# fit_eq='$y = ax + b$: $a$='+ fit_param[0] +', $b$=' + fit_param[1]
# print 'fit parameters: a*veff*ln(veff)+b:'  #approx
# print popt
# plt.plot(x,y,'b--',label='fit'+fit_eq)
# graph.addtext(fig=fig1, subplot=211, text=fit_eq, x=280,y=100,fontsize=10,color='b')
# #------------------------------------------------------------------------------------






####5mm
graph.skipcolor(6)
figsize=(10,12)
fig5, ax5, color_patch5 = graph.errorfill(veff5_tr_eight_masked,vring5_tr_eight_masked, vring5_tr_eight_err, marker='None', linewidth=2, linestyle='None',
                 label='L/D=1.32, top right', markersize=10,  subplot=212, figsize=figsize)  # span=5mm
fig6, ax6, color_patch6 = graph.errorfill(veff5_tl_eight_masked,vring5_tl_eight_masked, vring5_tl_eight_err, marker='None', linewidth=2, linestyle='None',
                 label='L/D=1.32, top left', markersize=10,  subplot=212, figsize=figsize)  # span=5mm
fig7, ax7, color_patch7 = graph.errorfill(veff5_br_eight_masked,vring5_br_eight_masked, vring5_br_eight_err, marker='None', linewidth=2, linestyle='None',
                 label='L/D=1.32, bottom right', markersize=10,  subplot=212, figsize=figsize)  # span=5mm
fig8, ax8, color_patch8 = graph.errorfill(veff5_bl_eight_masked,vring5_bl_eight_masked, vring5_bl_eight_err, marker='None', linewidth=2, linestyle='None',
                 label='L/D=1.32, bottom left', markersize=10,  subplot=212, figsize=figsize)  # span=5mm

graph.labelaxes(xlabel='$\overline{v_p^2}$ / ${\overline{v_p}}$ [mm/s]',ylabel='vortex ring speed [mm/s]', fontsize=10)
plt.legend(handles=[color_patch5, color_patch6, color_patch7, color_patch8])
graph.setaxes(ax5, xmin, xmax, ymin, ymax)

#
# #Add fit curve: a(xln(x)+b)-----------------------------------------
# popt, pcov = curve_fit(func3, veff5_tr_eight_masked[:-1],vring5_tr_eight_masked[:-1], bounds=([0,-100], [1. , 100]))
# x=np.arange(1, 500,5)
# y=func3(x,*popt)
#
# fit_param=list()
# for item in popt:
#     fit_param.append(str(round(item,4)))
# fit_eq='$y=a x (ln(x)+b)$: $a$='+ fit_param[0] +', $b$=' + fit_param[1]
# print 'fit parameters: a*veff*ln(veff)+b:'  #approx
# print popt
# plt.plot(x,y,'r--',label='fit'+fit_eq)
# graph.addtext(fig=fig1, subplot=212, text=fit_eq, x=320,y=350,fontsize=10,color='r')
#
# #------------------------------------------------------------------------------------
# #Add fit curve: a(xln(x)+b)-----------------------------------------
# veff5_tr_eight_masked_fit = veff5_tr_eight_masked[:-2]
# vring5_tr_eight_masked_fit = vring5_tr_eight_masked[:-2]
#
# popt, pcov = curve_fit(func2, veff5_tr_eight_masked_fit, vring5_tr_eight_masked_fit)
# x=np.arange(1, 500,5)
# y=func2(x,*popt)
#
# fit_param=list()
# for item in popt:
#     fit_param.append(str(round(item,4)))
# fit_eq='$y = ax + b$: $a$='+ fit_param[0] +', $b$=' + fit_param[1]
# print 'fit parameters: a*veff*ln(veff)+b:'  #approx
# print popt
# plt.plot(x,y,'b--',label='fit'+fit_eq)
# graph.addtext(fig=fig1, subplot=212, text=fit_eq, x=320,y=150,fontsize=10,color='b')
#------------------------------------------------------------------------------------


#plt.show()
savedir = "/Volumes/labshared3-1/takumi/good_data/vortex_ring_characterization/figures/"
filename = "vtop_vbottom_8rings"
filepath = savedir + filename
graph.save(filepath)



vring8_tr_eight_max = (vring8_tr_eight_masked + vring8_tr_eight_err)**2
vring8_tr_eight_min = (vring8_tr_eight_masked - vring8_tr_eight_err)**2
vring5_tr_eight_max = (vring5_tr_eight_masked + vring5_tr_eight_err)**2
vring5_tr_eight_min = (vring5_tr_eight_masked - vring5_tr_eight_err)**2

vring8_br_eight_max = (vring8_br_eight_masked + vring8_br_eight_err)**2
vring8_br_eight_min = (vring8_br_eight_masked - vring8_br_eight_err)**2
vring5_br_eight_max = (vring5_br_eight_masked + vring5_br_eight_err)**2
vring5_br_eight_min = (vring5_br_eight_masked - vring5_br_eight_err)**2

vring2total8 = np.mean(np.array([vring8_tr_eight_max + vring8_br_eight_max ,vring8_tr_eight_min + vring8_br_eight_min]),axis=0)
vring2total5 = np.mean(np.array([vring5_tr_eight_max + vring5_br_eight_max ,vring5_tr_eight_min + vring5_br_eight_min]),axis=0)


vratio8min_br = vring8_br_eight_min/ vring2total8
vratio8min_tr = vring8_tr_eight_min/ vring2total8
vratio8max_br = vring8_br_eight_max/ vring2total8
vratio8max_tr = vring8_tr_eight_max/ vring2total8

vratio8tr = np.mean(np.array([vratio8max_tr,vratio8min_tr]), axis=0)
vratio8br = np.mean(np.array([vratio8max_br,vratio8min_br]), axis=0)


#err
vratioerr8max_tr = vratio8max_tr - vratio8tr
vratioerr8min_tr = -vratio8min_tr + vratio8tr
vratioerr8_tr = np.array([vratioerr8max_tr,vratioerr8min_tr])
vratioerr8max_br = vratio8max_br - vratio8br
vratioerr8min_br = -vratio8min_br + vratio8br
vratioerr8_br = np.array([vratioerr8max_br, vratioerr8min_br])

###

vratio5min_br = vring5_br_eight_min/ vring2total5
vratio5min_tr = vring5_tr_eight_min/ vring2total5
vratio5max_br = vring5_br_eight_max/ vring2total5
vratio5max_tr = vring5_tr_eight_max/ vring2total5

vratio5tr = np.mean(np.array([vratio5max_tr,vratio5min_tr]),axis=0)
vratio5br = np.mean(np.array([vratio5max_br,vratio5min_br]),axis=0)
#err
vratioerr5max_tr = vratio5max_tr - vratio5tr
vratioerr5min_tr = -vratio5min_tr + vratio5tr
vratioerr5_tr = np.array([vratioerr5max_tr,vratioerr5min_tr])
vratioerr5max_br = vratio5max_br - vratio5br
vratioerr5min_br = -vratio5min_br + vratio5br
vratioerr5_br = np.array([vratioerr5max_br,vratioerr5min_br])

graph.skipcolor(6)
figsize=(8,14)
ymin, ymax = 0, 1.0
fig1, ax1, color_patch1 = graph.errorfill(veff8_tr_eight_masked,vratio8tr, vratioerr8_tr, marker='None', linewidth=2, linestyle='None',
                 label='L/D=2.11, top', markersize=10,  subplot=211, figsize=figsize,fignum=2)  # span=8mm
fig2, ax2, color_patch2 = graph.errorfill(veff8_tr_eight_masked,vratio8br, vratioerr8_br, marker='None', linewidth=2, linestyle='None',
                 label='L/D=2.11, bottom', markersize=10,  subplot=211, figsize=figsize,fignum=2)  # span=5mm

plt.legend(handles=[color_patch1, color_patch2])
graph.setaxes(ax1, xmin, xmax, ymin, ymax)
graph.labelaxes(xlabel='$\overline{v_p^2}$ / ${\overline{v_p}}$ [mm/s]',ylabel='$v_{i}^2/(v_{top}^2 + v_{bottom}^2)$', fontsize=10)


fig3, ax3, color_patch3 = graph.errorfill(veff5_tr_eight_masked,vratio5tr, vratioerr5_tr, marker='None', linewidth=2, linestyle='None',
                 label='L/D=1.32, top', markersize=10,  subplot=212, figsize=figsize,fignum=2)  # span=8mm
fig4, ax4, color_patch4 = graph.errorfill(veff5_tr_eight_masked,vratio5br, vratioerr5_br, marker='None', linewidth=2, linestyle='None',
                 label='L/D=1.32, bottom', markersize=10,  subplot=212, figsize=figsize,fignum=2)  # span=5mm

plt.legend(handles=[color_patch3, color_patch4])
graph.setaxes(ax3, xmin, xmax, ymin, ymax)
graph.labelaxes(xlabel='$\overline{v_p^2}$ / ${\overline{v_p}}$ [mm/s]',ylabel='$v_{i}^2/(v_{top}^2 + v_{bottom}^2)$', fontsize=10)

graph.show()
# savedir = "/Volumes/labshared3-1/takumi/good_data/vortex_ring_characterization/figures/"
# filename = "vfraction_8rings"
# filepath = savedir + filename
# graph.save(filepath)
