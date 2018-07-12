import os
import sys

def splitext(filename):
    return os.path.splitext(filename)

def get_parent_dir(filename):
    return os.path.dirname(filename)

def get_parent_dir_of_current_dir(dirname):
    return os.path.dirname(os.path.dirname(dirname))


def convert_decimalstr_to_float(str):
    """
    decimalstr is, for example, '5p3'. This method translates this string into 5.3 (float).

    Parameters
    ----------
    str: decimal str

    Returns
    -------
    number: float
    """
    return float(str.replace('p', '.'))

def convert_float_to_decimalstr(number):
    """
    decimalstr is, for example, '5p3'. This method translates this string into 5.3 (float).

    Parameters
    ----------
    number: decimal str

    Returns
    -------
    str: float
    """
    number_str = str(number)
    return str.replace('.', 'p')



def get_float_from_str(str, start, end):
    """
    Extract a number from a string like '_lalala_start5p3end_lalala' -> 5.3 (float)
    Parameters
    ----------
    str:
    start:
    end:

    Returns
    -------
    number: float

    """
    if str.find(start) < 0:
        print 'ERROR: ' + start + ' is not in ' + str
        raise NameError('issues with ' + start)
        #sys.exit(1)
    if str.find(end) < 0:
        print 'ERROR: ' + end + ' is not in ' + str
        raise NameError('issues with ' + end)
        #sys.exit(1)
    start_ind, end_ind = str.find(start), str.find(end)
    str_extracted = str[start_ind+len(start):end_ind]
    # print start_ind, end_ind
    # print str_extracted
    try:
        number = float(str_extracted)
        return number
    except ValueError:
        try:
            number = convert_decimalstr_to_float(str_extracted)
            return number
        except ValueError:
            print 'ERROR: ' + str_extracted + ' cannot be converted into float'
            #sys.exit(1)

def get_str_from_str(str, start, end):
    """
    Extract a string from a string like '_lalala_start5p3end_lalala' -> 5.3 (float)
    Parameters
    ----------
    str:
    start:
    end:

    Returns
    -------
    number: float

    """
    if str.find(start) < 0:
        print 'ERROR: ' + start + ' is not in ' + str
        raise NameError('issues with ' + start)
        #sys.exit(1)
    if str.find(end) < 0:
        print 'ERROR: ' + end + ' is not in ' + str
        raise NameError('issues with ' + end)
        #sys.exit(1)
    start_ind, end_ind = str.find(start), str.find(end)
    str_extracted = str[start_ind+len(start):end_ind]
    return str_extracted



