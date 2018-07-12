# -*- coding: utf-8 -*-
"""
Module for reading/writing a pickle
author: takumi
"""
import pickle
import os
import sys

def write(obj, filepath, verbose=True):
    """
    Generate a pickle file from obj
    Parameters
    ----------
    obj
    filepath
    verbose

    Returns
    -------

    """
    # Extract the directory and filename from the given path
    directory, filename = os.path.split(filepath)[0], os.path.split(filepath)[1]
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    pickle_out = open(filepath, "wb")
    pickle.dump(obj, pickle_out)
    if verbose:
        print 'Saved data under ' + filepath
    pickle_out.close()

def read(filename):
    pickle_in = open(filename, "rb" )
    obj = pickle.load(pickle_in)
    return obj

###################
# Codes below were written by Stephane

def write_s(obj,filename):
    
    try:
        f=open(filename,'wb')
    except 'EOFError':
        print('Empty file')
        return None
    p = pickle.Pickler(f,2)
#        S_str=self.decode()
    try:
        p.dump(obj)    
    except '_pickle.PicklingError':
        print('Sdata class has been modified')
    f.close()    
    
def read_s(filename):
    if os.path.isfile(filename):
       # print("Reading Sdata")# from "+filename)
        #has to be secured with a unic identifier.
        #bad method : the filename is used to compare with the attributes previously used to generate it
        v=sys.version_info

        f=open(filename,'rb')
        if v[0]==3:
            buf=f.read()
        
            S=pickle.loads(buf,encoding='latin1')
        else:
            S=pickle.load(f)
            
        f.close()
        
        return S
    else:
        print(filename+ "does not exist")
        return None
    