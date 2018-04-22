"""Module to format dictionaries"""
import numpy as np

def make_dict_json_serializable(datadict):
    """
    Allowed types for json: str, int, float, dict, list, bool, None

    Parameters
    ----------
    datadict

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


