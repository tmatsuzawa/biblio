"""Module to format dictionaries"""
import numpy as np
import collections

def make_dict_json_serializable(datadict):
    """
    Make a dictionary JSON serializable. datadict accepts a multi-hierarchial dictionary
    Allowed types for json: str, int, float, dict, list, bool, None

    Parameters
    ----------
    datadict: dict
    data arrays are stored like... datadict[key][subkey][subsubkey][subsubsubkey] = data

    Returns
    -------

    """
    allowed_types = [str, int, float, dict, list, bool, None]
    for key in datadict:
        if type(datadict[key])==dict:
            for subkey in datadict[key]:
                if type(datadict[key][subkey]) == dict:
                    for subsubkey in datadict[key][subkey]:
                        if type(datadict[key][subkey][subsubkey]) == dict:
                            for subsubsubkey in datadict[key][subkey][subsubkey]:
                                if not type(datadict[key][subkey][subsubkey][subsubsubkey]) in allowed_types:
                                    datadict[key][subkey][subsubkey][subsubsubkey] = list(datadict[key][subkey][subsubkey][subsubsubkey])
                                for i in range(len(datadict[key][subkey][subsubkey][subsubsubkey])):
                                    if not type(datadict[key][subkey][subsubkey][subsubsubkey][i]) in allowed_types:
                                        try:
                                            datadict[key][subkey][subsubkey][subsubsubkey][i] = list(datadict[key][subkey][subsubkey][subsubsubkey][i])
                                        except:
                                            continue

                        else:
                            if not type(datadict[key][subkey][subsubkey]) in allowed_types:
                                datadict[key][subkey][subsubkey] = list(datadict[key][subkey][subsubkey])
                            for i in range(len(datadict[key][subkey][subsubkey])):
                                if not type(datadict[key][subkey][subsubkey][i]) in allowed_types:
                                    try:
                                        datadict[key][subkey][subsubkey][i] = list(datadict[key][subkey][subsubkey][i])
                                    except:
                                        continue
                else:
                    if not type(datadict[key][subkey]) in allowed_types:
                        datadict[key][subkey] = list(datadict[key][subkey])
                    for i in range(len(datadict[key][subkey])):
                        if not type(datadict[key][subkey][i]) in allowed_types:
                            try:
                                datadict[key][subkey][i] = list(datadict[key][subkey][i])
                            except:
                                continue
        else:
            if not type(datadict[key]) in allowed_types:
                datadict[key] = list(datadict[key])
            for i in range(len(datadict[key])):
                if not type(datadict[key][i]) in allowed_types:
                    try:
                        datadict[key][i] = list(datadict[key][i])
                    except:
                        continue

    print '...Converted contents of a dictionary into lists'
    return datadict


def update_data_dict(dict, key, subkey, data=[]):
    """
    Generate a dictionary that stores effective velocity
    Parameters
    ----------
    dict
    key: span like span5.4
    subkey: commanded velocity, str
    data: effective velocity, float

    Returns
    -------

    """
    if not key in dict:
        dict[key] = {}  # Generate a sub-dictionary
    dict[key][subkey] = data
    return dict

def make_default_data_dict(keys_list):
    """
    Not quite functioning... why did i write this? 06/10/takumi
    Parameters
    ----------
    keys_list

    Returns
    -------

    """
    num_keys = len(keys_list)
    nested_dict = lambda: collections.defaultdict(nested_dict)
    datadict = nested_dict()
    if num_keys == 1:
        for i in range(num_keys):
            datadict[keys_list[0]] = np.nan
            # Make datadict a dictionary from defaultdict object
    elif num_keys == 2:
        for i in range(num_keys):
            datadict[keys_list[0]][keys_list[1]] = np.nan
    elif num_keys == 3:
        for i in range(num_keys):
            datadict[keys_list[0]][keys_list[1]][keys_list[2]] = np.nan
    elif num_keys == 4:
        for i in range(num_keys):
            datadict[keys_list[0]][keys_list[1]][keys_list[2]][keys_list[3]] = np.nan
    elif num_keys == 5:
        for i in range(num_keys):
            datadict[keys_list[0]][keys_list[1]][keys_list[2]][keys_list[3]][keys_list[4]] = np.nan
    elif num_keys > 5:
        print 'Currently, this method can NOT make a datadict with nestedness of more than 5.'
    else:
        print 'Error: Provided key_list is invalid. Returning None...'
        return None

    datadict = dict(datadict)
    if len(keys_list) >= 1:
        for key in datadict:
            datadict[key] = dict(datadict[key])
            if len(keys_list) >= 2:
                for subkey in datadict[key]:
                    datadict[key][subkey] = dict(datadict[key][subkey])
                    if len(keys_list) >= 3:
                        for subsubkey in datadict[key][subkey]:
                            datadict[key][subkey][subsubkey] = dict(datadict[key][subkey][subsubkey])
                            if len(keys_list) >= 4:
                                for subsubsubkey in datadict[key][subkey][subsubsubkey]:
                                    datadict[key][subkey][subsubkey][subsubsubkey] = dict(datadict[key][subkey][subsubkey][subsubsubkey])
                                    if len(keys_list) >= 5:
                                        for subsubsubsubkey in datadict[key][subkey][subsubkey][subsubsubkey]:
                                            datadict[key][subkey][subsubkey][subsubsubkey][subsubsubsubkey] = dict(datadict[key][subkey][subsubkey][subsubsubkey][subsubsubsubkey])
    return datadict
