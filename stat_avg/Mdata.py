
import stephane.manager.access as access
import numpy as np

import stephane.analysis.vgradient as vgradient

def set_parameters():
    """
    Return a dictionnary containing the parameters to save.
    Can be copied from a father Mdata ?"""
    
    List_key = ['t','x','y','Ux','Uy','Uz','E','omega','strain']
#    for k in List_key:    
    return List_key

def param_heritance(M,typ = 'avg'):
    """
    Return a dictionnary containing the parameters to save.
    Can be copied from a father Mdata ?"""
    
    names = ['t','x','y','param','id','dataDir','Sdata']#,'omega','strain']
    
    param = {}
    for name in names:
        param[name] = access.get_attr(M,name)
    
    param['type'] = typ
    return param

def ensemble_average(Mlist,field):
    U_moy = []
    
    for M in Mlist:
        U_moy.append(access.get_all(M,field))
        #(M,field,t,Dt=1,filter=False,Dt_filt=1,compute=False))

    nt = [U.shape[2] for U in U_moy]
    NT = min(nt)
    
    U_moy=[U[...,0:NT] for U in U_moy]
        
#    for i in range(10):
#        print(U_moy[i].shape)
    U_moy = np.nanmean(np.asarray(U_moy),axis=0)   
    
    print(U_moy.shape) 
    return U_moy
    
def make(Mlist):
    fields = ['Ux','Uy','E','dU']
    M_ref = Mlist[0]
    param = param_heritance(Mlist[0])  # herits the parameter of the first run
    
    data = {}
    for field in fields:
        if hasattr(M_ref,field):
            print('Compute average of '+field+' ...')
            data[field] = ensemble_average(Mlist,field)
        else :
            print('Compute average of '+field+' ...')
            print('Compute values on each field first')
            data[field] = ensemble_average(Mlist,field)
    
    return data,param
    
class Mdata:
    def __init__(self,Mlist,N=None):
        data,param = make(Mlist)
        
        self.load_default(Mlist)
        
        self.load(data)
        self.load(param)
    
    def load_default(self,Mlist):
        M = Mlist[0]
        for key in M.__dict__.keys():
            setattr(self,key,M.__dict__[key])
     #   self.filename = 
    
    def load(self,d):
        for name in d.keys():
            setattr(self,name,d[name])
        
    def shape(self):
        return self.Ux.shape
        
########################## Measurements #############################

    def get(self,field,**kwargs):
        
        if field == 'U':
            #return both component in a vectorial format
            Ux = self.get('Ux')
            Uy = self.get('Uy')
            data = np.transpose(np.asarray([Ux,Uy]),(1,2,3,0))
            return data
#        if (not hasattr(self,field)) or (compute):
            
#            vgradient.compute(M,field,Dt=Dt_filt,**kwargs)        
        if not hasattr(self,field):
            if 'Dt_filt' in kwargs and kwargs['Dt_filt']>1:
                print('Filtering of the data : irreversible')
            self.compute(field)
#            setattr(self,field,)
        if hasattr(self,field):
            return getattr(self,field)
        else:
            return None
    
    def get_cut(self):
        print("crazy")
        
    def compute(self,field,**kwargs):
        return vgradient.compute(self,field,**kwargs)

    def measure(self,name,function,force=False,*args,**kwargs):
        if (not hasattr(self,name)) or force:
            print("Compute "+name)
            val = function(self,*args,**kwargs)
            setattr(self,name,val)
        else:
            print("Already computed")