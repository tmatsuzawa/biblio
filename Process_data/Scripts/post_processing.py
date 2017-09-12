

"""
Load a serie of Mdata, removes spurious vectors based on kinetic energy gradient
"""

import stephane.mdata.M_manip as M_manip
import stephane.analysis.cdata as cdata
import numpy as np
import stephane.jhtd.strain_tensor as strain

import stephane.display.graphes as graphes

import scipy.ndimage.filters as filters

import stephane.vortex.track as track
import os.path
import glob
import stephane.hdf5.h5py_convert as to_hdf5


def example(filename):
    figs = {}
    savedir = '/Users/stephane/Documents/Postdoc_Chicago/Process_data/Scripts/Example/Smoothing/'
    
    M = M_manip.load_Mdata_file(filename,data=True)
    M.get('E')
    M.get('omega')
    
    nx,ny,nt = M.shape()
    x,y = M.x,M.y
    
    i = nt/2
    
    # display raw energy
    graphes.Mplot(M,'E',i,vmin=0,vmax=50000,fignum=1,colorbar=True)
    figs.update(graphes.legende('$x$ (mm)','$y$ (mm)','Energy without pre-processing',cplot=True))
        
    #Display raw vorticity
    graphes.Mplot(M,'omega',i,vmin=0,vmax=80,fignum=2,colorbar=True)
    figs.update(graphes.legende('$x$ (mm)','$y$ (mm)','Vorticity without pre-processing',cplot=True))

    # Look for bada data point in amplitude
    E = M.E[...,i]
    tup,errors = cdata.rm_baddata(E,borne=2.)                              
    xc = []
    yc = []
    for t in tup:
        xc.append([x[t[0],t[1]]])
        yc.append([y[t[0],t[1]]])
    graphes.graph(xc,yc,label='ro',fignum=2)
    print(figs)
    graphes.save_figs(figs,savedir=savedir)

    Ux,Uy = M.Ux[...,i],M.Uy[...,i]
    #replace component of the velocity
    Ux = cdata.replace_data(Ux,tup)[...,0]
    Uy = cdata.replace_data(Uy,tup)[...,0]
    U = np.transpose(np.asarray([Uy,Ux]),(1,2,0))

    dU = strain.strain_tensor_C(U,b=2.)
    omega,enstrophy = strain.vorticity(dU,d=2,norm=False)
    E = Ux**2+Uy**2

    graphes.color_plot(x,y,omega,vmin=0,vmax=80,fignum=4)
    graphes.plt.colorbar()
    figs.update(graphes.legende('$x$ (mm)','$y$ (mm)','Vorticity after pre-processing',cplot=True))

    graphes.color_plot(x,y,E,vmin=0,vmax=50000,fignum=3)
    graphes.plt.colorbar()
    figs.update(graphes.legende('$x$ (mm)','$y$ (mm)','Energy after pre-processing',cplot=True))
    
    
    ### Gaussian filter
    omega_filt = filters.gaussian_filter(omega, sigma=2.)#[...,0]
    graphes.color_plot(x,y,omega_filt,vmin=0,vmax=80,fignum=5)
    graphes.plt.colorbar()
    figs.update(graphes.legende('$x$ (mm)','$y$ (mm)','Vorticity after gaussian smoothing',cplot=True))
    graphes.save_figs(figs,savedir=savedir)


def single(filename,display=False):
    figs = {}
    savedir = '/Users/stephane/Documents/Postdoc_Chicago/Process_data/Scripts/Samples/Smoothing/'
    
    filesave,ext = os.path.splitext(filename)
    filesave = filesave + '_processed.hdf5'
        
    M = M_manip.load_Mdata_file(filename,data=True)
    M.get('E')
    M.get('omega')
    
    nx,ny,nt = M.shape()
    i = nt/2
    
    # display raw energy
    if display:
        graphes.plt.close('all')        
        graphes.Mplot(M,'E',i,vmin=0,vmax=50000,fignum=1,colorbar=True)
        figs.update(graphes.legende('$x$ (mm)','$y$ (mm)','Energy without pre-processing',cplot=True))
        
    #Display raw vorticity
        graphes.Mplot(M,'omega',i,vmin=0,vmax=100,fignum=2,colorbar=True)
        figs.update(graphes.legende('$x$ (mm)','$y$ (mm)','Vorticity without pre-processing',cplot=True))
        graphes.save_figs(figs,savedir=savedir,prefix=M.Id.get_id())

    
        graphes.Mplot(M,'Ux',i,vmin=-200,vmax=200,fignum=3,colorbar=True)
        figs.update(graphes.legende('$x$ (mm)','$y$ (mm)','Ux after pre-processing',cplot=True))

    M = cdata.remove_spurious(M,borne=1.5)
    M.get('E',force=True)
    M.get('omega',force=True)
    
    if display:
        graphes.Mplot(M,'E',i,vmin=0,vmax=50000,fignum=4,colorbar=True)
        figs.update(graphes.legende('$x$ (mm)','$y$ (mm)','Energy after pre-processing',cplot=True))
        
    #Display raw vorticity
        graphes.Mplot(M,'omega',i,vmin=0,vmax=100,fignum=5,colorbar=True)
        figs.update(graphes.legende('$x$ (mm)','$y$ (mm)','Vorticity after pre-processing',cplot=True))
    
    ### Gaussian filter        
        omega_filt = track.smoothing(M,i,field='omega',sigma=2.5)
        graphes.color_plot(M.x,M.y,omega_filt,vmin=0,vmax=100,fignum=6)
        graphes.plt.colorbar()
        figs.update(graphes.legende('$x$ (mm)','$y$ (mm)','Vorticity after gaussian smoothing',cplot=True))
        graphes.save_figs(figs,savedir=savedir,prefix=M.Id.get_id())
    ### Gaussian filter
#    omega_filt = filters.gaussian_filter(omega, sigma=2.)#[...,0]
#    graphes.color_plot(x,y,omega_filt,vmin=0,vmax=50,fignum=5)
#    graphes.plt.colorbar()
#    figs.update(graphes.legende('$x$ (mm)','$y$ (mm)','Vorticity after gaussian smoothing',cplot=True))
#   graphes.save_figs(figs,savedir=savedir)


    to_hdf5.write(M,filename=filesave,key='Mdata')
#    M.write(data=True,overwrite=True)

def multiple(folder):
    savedir = '/Users/stephane/Documents/Postdoc_Chicago/Process_data/Scripts/Samples/Smoothing/'
    
    fileList = glob.glob(folder+'*1.hdf5')
    for filename in fileList:    
        print(filename)
        filesave,ext = os.path.splitext(filename)
        filesave = filesave + '_processed.hdf5'

        if os.path.isfile(filesave):
            print('already done')
        else:
            single(filename,display=True)
    
def main():
    folder = '/Users/stephane/Documents/Postdoc_Chicago/Experimental_data/2017_05_16/Mdata_W64pix/'
    multiple(folder)
    filename = folder + 'M_2017_05_16_0_0.hdf5'
    #filename = '/Users/stephane/Documents/Postdoc_Chicago/Experimental_data/2017_05_16/Mdata/M_2017_05_16_0_0_processed.hdf5'
    #example(filename)    
    #single(filename)
    
main()