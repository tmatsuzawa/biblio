'''
Module for organizing data from txt files for various computations
'''
import numpy as np
import os
import matplotlib.pyplot as plt


#cwd = os.getcwd()  # get the path of the current working directory
#filelist = os.listdir(cwd)  # list files in the directory

datafile='circulation_fv_fps4000_left_macro105mm_D20mm_spanVaried_freq0.2Hz_vVaried.txt'
data_path='C:\Users\labusr\Documents\git\circulation_fv_fps4000_left_macro105mm_D20mm_spanVaried_freq0.2Hz_vVaried.txt'

def generate_data_arrays(datafile, data_path):
    """
    Read data from txt files, and make an array
    Parameters
    ----------
    datafile : string
        the name of the data txt file
    data_path : string
        the path of the data txt file

    radius : float or None
        the radius of a disc to examine, if not None. If region is not None, radius is ignored
    polygon : #vertices x 2 numpy float array
        If not none, use this closed path to define the region of interes
    display : bool
        show some intermediate results to check that code is functioning
    dt : int (default=10)
        Number of frames to smooth over
    center_frame : bool
        Cut off the edges of the frame, keeping the center within distance 'radius' of center pixel rather than using
        mm.x and mm.y for position information

    Returns
    -------
    s_k :
    k :
    """
    var1=list()
    var2=list()
    var3 = list()
    var4 = list()
    var5 = list()
    var6 = list()
    var7 = list()
    var8 = list()
    var9 = list()
    var10 = list()
    var11 = list()
    var12 = list()
    var13 = list()
    var14 = list()
    var15 = list()
    var16 = list()
    var17 = list()
    var18 = list()
    var19 = list()
    var20 = list()

    f = open(data_path, 'r')
    counter=1

    for line in f:
        if counter==1:
            key = line.split()
        if counter==2:
            val = line.split()
            break
        counter = counter+1

    param = dict(zip(key, val))

    # Find keys whose values cannot be converted to float
    bad_key = list()
    for key in param:
        try:
            param[key] = float(param[key])
        except ValueError:
            # print param[key] + ' cannot be converted to float.'
            bad_key.append(key)

    # Convert strings into float if possible
    for key in param:
        if not key in bad_key:
            param[key] = float(param[key])





    for line in f:
        if counter==1:
            variables = line.split()
            # for x in range(0,len(variables)):
            #     dct["var{0}".format(x)]=

        else:
            elements = line.split()
            for i in range(0,len(variables)):
                try:
                    elements[i]=float(elements[i]

                except ValueError:
                    continue


    f.close()
    return variables


def addition(x, y):
    z=x+y
    return z