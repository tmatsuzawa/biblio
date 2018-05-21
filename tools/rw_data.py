"""
Module for read/write
@author: takumi
"""

import os
import sys
import numpy as np
import json
import csv
import pickle
import h5py
import library.basics.formatdict as fd

## Read
# json
def read_json(datafilepath, verbose=True):
    data = json.load(open(datafilepath))
    if verbose:
        print 'Data was successfully loaded from ' + datafilepath
    return data

#pickle
def read_pickle(filename):
    pickle_in = open(filename, "rb" )
    obj = pickle.load(pickle_in)
    return obj
# csv
def reader_csv(datafilepath):
    """
    Returns a (csv) reader object from a csv file
    Parameters
    ----------
    datafilepath

    Returns
    -------
    csvreader: csv reader object
    """
    with open(datafilepath, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#        for row in spamreader:
#            print ', '.join(row)
    return csvreader

# More general data reader; default is csv.
def read_data(datafilepathpath, delimiter=','):
    """
    Versatile method to read data
    Parameters
    ----------
    datafilepathpath
    delimiter

    Returns
    -------
    data : numpy array

    """
    data = np.genfromtxt(datafilepathpath, delimiter=delimiter)
    return data


## Write
def write_json(datafilepath, datadict):
    """
    Generates a json file from a dictionary (Formerly named as save_dict_to_json)
    Parameters
    ----------
    datafilepath
    datadict

    Returns
    -------

    """
    with open(datafilepath, 'w') as fyle:
        try:
            json.dump(datadict, fyle, sort_keys=True, indent=1, separators=(',', ': '))
            fyle.close()
        except TypeError:
            datadict = fd.make_dict_json_serializable(datadict)
            json.dump(datadict, fyle, sort_keys=True, indent=1, separators=(',', ': '))
            fyle.close()
    print 'Data was successfully saved as ' + datafilepath

def write_pickle(filepath, obj, verbose=True):
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


def write_hdf5_dict(filepath, data_dict):
    """
    Stores data_dict
    Parameters
    ----------
    filepath :  str
                file name where data will be stored. (Do not include extension- .h5)
    data_dict : dictionary
                data should be stored as data_dict[key]= data_arrays

    Returns
    -------

    """
    ext = '.h5'
    filename = filepath + ext
    hf = h5py.File(filename, 'w')
    for key in data_dict:
        hf.create_dataset(key, data=data_dict[key])
    hf.close()
    print 'Data was successfully saved as ' + filename

def write_hdf5_simple(filepath, x, y):
    """
    Stores data_dict
    Parameters
    ----------
    filepath :  str
                file name where data will be stored. (Do not include extension- .h5)
    data_dict : dictionary
                data should be stored as data_dict[key]= data_arrays

    Returns
    -------

    """
    data_dict = {}
    data_dict['x'] = x
    data_dict['y'] = y

    ext = '.h5'
    filename = filepath + ext
    hf = h5py.File(filename, 'w')
    for key in data_dict:
        hf.create_dataset(key, data=data_dict[key])
    hf.close()
    print 'Data was successfully saved as ' + filename


