import os
import sys

def get_filename_wo_ext(filename):
    filename_no_ext, ext = os.path.splitext(filename)
    filename_short = os.path.split(filename_no_ext)
    return filename_short[1]

def splitext(filename):
    """
        /full/path/to/.../file.ext
    ->  [/full/path/to/..., file.ext]
    Parameters
    ----------
    filename

    Returns
    -------

    """
    return os.path.split(filename)

def get_parent_dir(filename):
    return os.path.dirname(filename)

def get_parent_dir_of_current_dir(dirname):
    return os.path.dirname(os.path.dirname(dirname))


def convert_decimalstr_to_float(str):
    """
    Convert a decimal string into float.
    E.g. "5p3" -> 5,3
    decimalstr is, for example, '5p3'. This method translates this string into 5.3 (float).

    Parameters
    ----------
    str: decimal str

    Returns
    -------
    number: float
    """
    return float(str.replace('p', '.'))

def convert_float_to_decimalstr(number, zpad=3):
    """
    Convert float into decimal string which replaces a decimal point with a letter p.
    e.g. 5.3 -> "5p3"

    Parameters
    ----------
    number: decimal str

    Returns
    -------
    str: string
    """
    number_str = str(number)
    number_str = number_str.replace('.', 'p')
    return number_str.rjust(zpad, '0')



def get_float_from_str(str, start, end):
    """
    Extract a number from a string like '_lalala_start5p3end_lalala' -> 5.3 (float)
    Automatically convert to float if decimalstr (ex. 2p6 for 2.6) is given

    Parameters
    ----------
    str: str
    start: str
    end: str

    Returns
    -------
    number: float

    """
    if str.find(start) < 0 and start != '':
        print 'ERROR: ' + start + ' is not in ' + str
        raise NameError('issues with ' + start)
        #sys.exit(1)
    if str.find(end) < 0 and end != '':
        print 'ERROR: ' + end + ' is not in ' + str
        raise NameError('issues with ' + end)
        #sys.exit(1)
    if start == '':
        start_ind = 0
    else:
        start_ind = str.find(start)

    new_str = str[start_ind + len(start):]
    if end == '':
        end_ind = len(new_str)
    else:
        end_ind = new_str.find(end)
    # str_extracted = str[start_ind+len(start):end_ind]
    str_extracted = new_str[:end_ind]
    try:
        number = float(str_extracted)
        return number
    except ValueError:
        try:
            number = convert_decimalstr_to_float(str_extracted)
            return number
        except ValueError:
            print 'ERROR: ' + str_extracted + ' cannot be converted into float'
            raise RuntimeError

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



def latex_float(number, hideone=True):
    """
    Returns a string in a scientific format (latex), base 10
    Parameters
    ----------
    number: float ... e.g. 52481
    hideone: bool ... If hideone is True and the result becomes "1 x 10^m", return only "10^m".

    Returns
    -------
    float_str: string ... e.g. "5 x 10^4"

    """
    float_str = "{0:.2g}".format(number)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        if int(round(float(base))) == 1:
            return r"$10^{{{0}}}$".format(int(exponent))
        else:
            return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))

    else:
        return float_str