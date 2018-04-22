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

def write_pickle(obj, filepath, verbose=True):
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


