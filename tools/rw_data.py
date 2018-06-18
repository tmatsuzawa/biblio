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
def read_csv(datapath, encoding='utf-8-sig'):
    """
    Returns data_name (1st line of a csv file) as a list and data as a 2d array

    Parameters
    ----------
    datapath: str, location of csv data

    Returns
    -------
    data_name: list
    data: 2d array

    """
    from io import open
    with open(datapath, 'rb') as csvfile:
        # If the csv file contains UTF-8-BOM in the beginning, make it unicode
        #
        # csvfile = csvfile.read().decode("utf-8-sig").encode("utf-8")
        # print  type(csvfile.read())

        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        data = []
        for i, row in enumerate(reader):
            if i == 0:
                data_names = [s.decode(encoding).encode('ascii') for s in row]
            else:
                data.append([float(s.decode(encoding).encode('ascii')) for s in row])
        data = np.array(data).reshape(i, len(data_names))

    return data_names, data

# A general data reader; default is csv.
#   - must contain ONLY data no strings
def read_data(datapath, delimiter=',', skiprows=0,  **kwargs):
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
    data = np.loadtxt(datapath, delimiter=delimiter, skiprows=skiprows, **kwargs)
    return data

# read data and return a data_dict
def make_data_dict_from_csv(datapath, key_from_data_name, subkey_from_data_name, data_from_data_name, **kwargs):
    """
    Assumes that the data is stored in the following format
    x,  y,  z, ...
    0.1, -9.2, 2.3, ...
    8.1, -2.2, 5.3, ...
    Parameters
    ----------
    datapath
    delimiter
    skiprows
    kwargs

    Returns
    -------

    """
    datadict = {}
    data_names, data = read_csv(datapath)
    for data_name in data:
        key = key_from_data_name + str()
        datadict = fd.update_data_dict(datadict, key, subkey, data)




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


