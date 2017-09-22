'''
Module for organizing data from txt files for various computations
'''
import numpy as np
import os
import matplotlib.pyplot as plt


#cwd = os.getcwd()  # get the path of the current working directory
#filelist = os.listdir(cwd)  # list files in the directory

def generate_data_dct(data_path,separation=' '):
    """
    Read data from txt files, and make a dictionary
    Parameters
    ----------
    data_path : string
        absolute path of the data txt file

    Returns
    -------
    data : dictionary
        keys are named as var0, var1, ...var#. Corresponding values are lists.
    """
    f = open(data_path, 'r')
    counter=1
    data={}
    for line in f:
        if counter==1:
            key = line.split(separation)
            for x in range(0,len(key)):
                data["var{0}".format(x)] = []  # initialize lists var0, var1,...

        if not counter==1:
            val = line.split(separation)
            for i,x in enumerate(val):
                try:
                    val[i]=float(val[i])
                except ValueError:
                    pass
            #print val
            #print len(key)
            for x in range(0, len(key)):
                data["var{0}".format(x)].append(val[x])
            #print data

        counter = counter+1
    return key, data,counter
