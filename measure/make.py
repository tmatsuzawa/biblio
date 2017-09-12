

import stephane.tools.rw_data as rw_data
import stephane.tools.browse as browse
import stephane.pprocess.check_piv as check
import stephane.mdata.Mdata_PIVlab as pivlab
import stephane.mdata.Sdata_manip as Sdata
import stephane.analysis.cdata as cdata
import numpy as np
import stephane.pprocess.test_serie as tests
import stephane.display.graphes as graphes

import stephane.analysis as analysis

import stephane.display.panel as panel


def define_dictionnary():
    
    pass

def single(M,fun,**args,**kwargs):
    """
    call a measurement function fun and generate the following output 
    """
    data = {}
    output = fun(M,**args,**kwargs)
    data['X'] = output[0]
    data['Y'] = output[1]
    data['param'] = output[2:]

    figs = {}
    graphes.graph(X,Y,fignum=1)
    figs.update(graphes.legende('$t$ (s)','$<E>_{x,y}$ (mm^2/s^2)',graphes.title(M)))

    savedir = location(M)
    graphes.save_figs(figs,savedir=savedir,suffix='label',prefix='',frmt='pdf',dpi=300,display=True)
    
    write_dictionnary(file,keys,List_info,delimiter='\t')
    
    return data
    
def dict_fun():
    
    functions = [decay.decay,]
    
def multiple(Mlist):
    pass
    
def location(M):
    savedir = './Vortex_Turbulence/Mean_flow/'+date+'/'
    
    
def panel_T():
    subplot = [int('24'+str(i)) for i in range(1,9)]#[131,132,133]
    
    
    
    
    
    
    
def example(M):

    single(M,analysis.decay.decay,display=False,label='',fignum=1,compute=True,fluctuations=False)

