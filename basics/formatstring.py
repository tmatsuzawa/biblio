import os
import sys

def splitext(filename):
    return os.path.splitext(filename)

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
    if str.find(end) < 0:
        print 'ERROR: ' + end + ' is not in ' + str
        sys.exit()
    start_ind, end_ind = str.find(start), str.find(end)
    str_extracted = str[str.find(start)+1, str.find(end)]
    try:
        number = float(str_extracted)
        return number
    except ValueError:
        try:
            number = convert_decimalstr_to_float(str_extracted)
            return number
        except ValueError:
            print 'ERROR: ' + str_extracted + ' cannot be converted into float'
            sys.exit()


