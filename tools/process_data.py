"""
Module to process multidimensional numpy arrays (cleaning and interpolation)
Numpy Masking module is heavily implemented in this module.
- author: takumi
"""
import numpy as np
import copy
import numpy.ma as ma
from scipy import interpolate

def interpolate_using_mask(arr1, mask):
    """
    Conduct linear interpolation for data points where its mask value is True

    ... This interpolation is not ideal because this flattens multidimensional array first, and takes a linear interpolation
    for missing values. That is, the interpolated values at the edges of the multidimensional array are nonsense b/c
    actual data does not have a periodic boundary condition.

    Parameters
    ----------
    arr1 : array-like (n x m), float
        array with unphysical values such as nan, inf, and ridiculously large/small values
        assuming arr1 is your raw data
    mask : array-like (n x m), bool

    Returns
    -------
    arr : array-like (n x m), float
        array with unphysical values replaced by appropriate values
    """
    arr2T = copy.deepcopy(arr1).T

    f0 = np.flatnonzero(mask)
    f1 = np.flatnonzero(~mask)

    arr1[mask] = np.interp(f0, f1, arr1[~mask])

    f0 = np.flatnonzero(mask.T)
    f1 = np.flatnonzero(~mask.T)
    arr2T[mask.T] = np.interp(f0, f1, arr1.T[~(mask.T)])
    arr2 = arr2T.T

    arr = (arr1 + arr2) * 0.5
    return arr


def get_mask_for_unphysical(U, cutoffU=2000., fill_value=99999., verbose=True):
    """

    Parameters
    ----------
    U: array-like
    cutoffU: float
        if |value| > cutoff, this method considers those values unphysical.
    fill_value:


    Returns
    -------
    mask: multidimensional boolean array

    """
    if verbose:
        print '...Note that nan/inf values in U are replaced by ' + str(fill_value)
        print '...number of invalid values (nan and inf) in the array: ' + str(np.isnan(U).sum() + np.isinf(U).sum())
        print '...number of nan values in U: ' + str(np.isnan(U).sum())
        print '...number of inf values in U: ' + str(np.isinf(U).sum()) + '\n'

    # Replace all nan and inf values with fill_value.
    # fix_invalid still enforces a mask on elements with originally invalid values
    U_fixed = ma.fix_invalid(U, fill_value=fill_value)
    n_invalid = ma.count_masked(U_fixed)
    if verbose:
        print '...number of masked elements by masked_invalid: ' + str(n_invalid)
    # Update the mask to False (no masking)
    U_fixed.mask = False



    # Mask unreasonable values of U_fixed
    b = ma.masked_greater(U_fixed, cutoffU)
    c = ma.masked_less(U_fixed, -cutoffU)
    n_greater = ma.count_masked(b) - n_invalid
    n_less = ma.count_masked(c)
    if verbose:
        print '...number of masked elements greater than cutoff: ' + str(n_greater)
        print '...number of masked elements less than -cutoff: ' + str(n_less)

    # Generate a mask for all nonsense values in the array U
    mask = ~(~b.mask * ~c.mask)

    d = ma.array(U_fixed, mask=mask)
    n_total = ma.count_masked(d)
    # U_filled = ma.filled(d, fill_value)

    #Total number of elements in U
    N = 1
    for i in range(len(U.shape)):
        N *= U.shape[i]
    print '...total number of unphysical values: ' + str(ma.count_masked(d)) + '  (' + str((float(n_total)/N*100)) + '%)\n'
    return mask



def fill_unphysical_with_sth(U, mask, fill_value=np.nan):
    """
    Returns an array whose elements are replaced by fill_value if its mask value is True
    Parameters
    ----------
    U   array-like
    mask   multidimensional boolean array
    fill_value   value that replaces masked values

    Returns
    -------
    U_filled  array-like

    """
    U_masked = ma.array(U, mask=mask)
    U_filled = ma.filled(U_masked, fill_value)  # numpy array. This is NOT a masked array.

    return U_filled

def get_mask_for_unphysical_using_median(U, cutoffratio=0.4, mode='less'):
    median = np.median(U)
    if mode=='less' or mode=='l':
        U_masked = ma.masked_less(U, median*cutoffratio)
    elif mode=='lesseqal' or mode=='leq':
        U_masked = ma.masked_less_equal(U, median * cutoffratio)
    elif mode=='greater' or mode=='g':
        U_masked = ma.masked_greater(U, median * cutoffratio)
    elif mode=='greaterequal' or mode=='geq':
        U_masked = ma.masked_greater_equal(U, median * cutoffratio)
    return U_masked.mask

def get_mask_for_unphysical_using_cutoff(U, cutoff=None, mode='less'):
    if mode=='less' or mode=='l':
        U_masked = ma.masked_less(U, cutoff)
    elif mode=='lesseqal' or mode=='leq':
        U_masked = ma.masked_less_equal(U, cutoff)
    elif mode=='greater' or mode=='g':
        U_masked = ma.masked_greater(U, cutoff)
    elif mode=='greaterequal' or mode=='geq':
        U_masked = ma.masked_greater_equal(U, cutoff)
    return U_masked.mask



def clean_vdata(M, cutoffU=2000, fill_value=np.nan, verbose=True):
    print 'Cleaning M.Ux...'
    mask = get_mask_for_unphysical(M.Ux, cutoffU=cutoffU, fill_value=fill_value, verbose=verbose)
    Ux_filled_with_nans = fill_unphysical_with_sth(M.Ux, mask, fill_value=fill_value)
    Ux_interpolated = interpolate_using_mask(Ux_filled_with_nans, mask)
    M.Ux[:]= Ux_interpolated[:]
    print 'Cleaning M.Uy...'
    mask = get_mask_for_unphysical(M.Uy, cutoffU=cutoffU, fill_value=fill_value, verbose=verbose)
    Uy_filled_with_nans = fill_unphysical_with_sth(M.Uy, mask, fill_value=fill_value)
    Uy_interpolated = interpolate_using_mask(Uy_filled_with_nans, mask)
    M.Uy[:]= Uy_interpolated[:]
    print '...Cleaning Done.'
    return M


## CLEANING N-D arrays
def clean_multi_dim_array(data, cutoff, verbose=True):
    """

    Parameters
    ----------
    data: N-d array
    cutoff: cutoff value above which will be removed and then interpolated
            elements below -cutoff will be also removed and then interpolated, float
    verbose: If True, prints out the number of elements that get removed and interpolated in details, bool

    Returns
    -------
    data_interpolated: N-d array with no np.nan, np.inf, any values above cutoff nor below -cutoff

    """
    mask = get_mask_for_unphysical(data, cutoffU=cutoff, fill_value=99999., verbose=verbose)
    data_filled_with_nans = fill_unphysical_with_sth(data, mask, fill_value=np.nan)
    data_interpolated = interpolate_using_mask(data_filled_with_nans, mask)
    '...Cleaning Done.'
    return data_interpolated

def clean_multi_dim_array_using_median(data, cutoffratio=0.4, mode='less'):
    """
    Cleans a multidimensional array using median
    Often, data comes with an array of arguments. e.g. x=[10.2, 20.4, np.nan, 40.7], t=[1., 2., 3., 4.]
    Then, this outputs x_clean=[10.2, 20.4, 40.7], t=[1., 2., 4.]

    Parameters
    ----------
    arg: N-d array argument array
    data: N-d array
    cutoffratio: ratio used to
    verbose

    Returns
    -------
    clean_arg N-d array
    clean_data N-d array

    """
    mask = get_mask_for_unphysical_using_median(data, cutoffratio, mode)
    data_masked =  ma.array(data, mask=mask)
    clean_data = data_masked.compress()
    '...Cleaning Done.'
    return clean_data, data_masked, mask

## CLEANING a N-D array PAIR
def clean_multi_dim_array_pair_using_median(arg, data, cutoffratio=0.4, mode='less'):
    """
    Cleans two multidimensional arrays using a mask of data
    Often, data comes with an array of arguments. e.g. x=[10.2, 20.4, np.nan, 40.7], t=[1., 2., 3., 4.]
    Then, this outputs x_clean=[10.2, 20.4, 40.7], t=[1., 2., 4.]

    Parameters
    ----------
    arg: N-d array argument array
    data: N-d array
    cutoffratio: ratio used to
    verbose

    Returns
    -------
    clean_arg N-d array
    clean_data N-d array

    """
    mask = get_mask_for_unphysical_using_median(data, cutoffratio, mode)
    data_masked =  ma.array(data, mask=mask)
    arg_masked = ma.array(arg, mask=mask)
    clean_data = data_masked.compressed()
    clean_arg = arg_masked.compressed()
    '...Cleaning Done.'
    return clean_arg, clean_data

def clean_multi_dim_array_pair_using_cutoff(arg, data, cutoff, mode='less'):
    """
    Cleans two multidimensional arrays using a mask of data
    Often, data comes with an array of arguments. e.g. x=[10.2, 20.4, np.nan, 40.7], t=[1., 2., 3., 4.]
    Then, this outputs x_clean=[10.2, 20.4, 40.7], t=[1., 2., 4.]

    Parameters
    ----------
    arg: N-d array argument array
    data: N-d array
    cutoffratio: ratio used to
    verbose

    Returns
    -------
    clean_arg N-d array
    clean_data N-d array

    """
    mask = get_mask_for_unphysical_using_cutoff(data, cutoff, mode)
    data_masked =  ma.array(data, mask=mask)
    arg_masked = ma.array(arg, mask=mask)
    clean_data = data_masked.compressed()
    clean_arg = arg_masked.compressed()
    '...Cleaning Done.'
    return clean_arg, clean_data

## CLEANING a N-D array TRIO
def clean_multi_dim_array_trio_using_median(arg, data1, data2, cutoffratio=0.4, mode='less'):
    """
    Cleans two multidimensional arrays using a mask of data
    Often, data comes with an array of arguments and error. e.g. x=[10.2, 20.4, np.nan, 40.7], xerr=[0.1, 0.3, np.nan,-0.6], t=[1., 2., 3., 4.]
    Then, this outputs x_clean=[10.2, 20.4, 40.7], xerr_clean=[0.1, 0.3,-0.6] ,t_clean=[1., 2., 4.]

    Parameters
    ----------
    arg: N-d array argument array
    data1: N-d array
    data2: N-d array
    cutoffratio: ratio used to
    verbose

    Returns
    -------
    clean_arg N-d array
    clean_data1 N-d array
    clean_data2 N-d array

    """
    mask = get_mask_for_unphysical_using_median(data1, cutoffratio, mode)
    data1_masked =  ma.array(data1, mask=mask)
    data2_masked =  ma.array(data2, mask=mask)
    arg_masked = ma.array(arg, mask=mask)
    clean_data1 = data1_masked.compressed()
    clean_data2 = data2_masked.compressed()
    clean_arg = arg_masked.compressed()
    '...Cleaning Done.'
    return clean_arg, clean_data1, clean_data2


def clean_multi_dim_array_trio_using_cutoff(arg, data1, data2, cutoff, mode='less'):
    """
    Cleans two multidimensional arrays using a mask of data
    Often, data comes with an array of arguments and error. e.g. x=[10.2, 20.4, np.nan, 40.7], xerr=[0.1, 0.3, np.nan,-0.6], t=[1., 2., 3., 4.]
    Then, this outputs x_clean=[10.2, 20.4, 40.7], xerr_clean=[0.1, 0.3,-0.6] ,t_clean=[1., 2., 4.]

    Parameters
    ----------
    arg: N-d array argument array
    data1: N-d array
    data2: N-d array
    cutoffratio: ratio used to
    verbose

    Returns
    -------
    clean_arg N-d array
    clean_data1 N-d array
    clean_data2 N-d array

    """
    mask = get_mask_for_unphysical_using_cutoff(data1, cutoff, mode)
    data1_masked =  ma.array(data1, mask=mask)
    data2_masked =  ma.array(data2, mask=mask)
    arg_masked = ma.array(arg, mask=mask)
    clean_data1 = data1_masked.compressed()
    clean_data2 = data2_masked.compressed()
    clean_arg = arg_masked.compressed()
    '...Cleaning Done.'
    return clean_arg, clean_data1, clean_data2

def interpolate_1Darrays(x, data, xint=None, xnum=None, mode='cubic'):
    """
    Conduct interpolation on a 1d array (N elements) to generate a 1d array (xnum elements)
    One can also specify x-spacing (xint) instead of the number of elements of the interpolated array
    Parameters
    ----------
    x
    data
    xint
    xnum
    mode

    Returns
    -------

    """
    xmin, xmax = np.min(x), np.max(x)
    if xint is None and xnum is None:
        xnum = len(x)*10
        xint = np.abs(xmax-xmin)/float(xnum)
    elif xint is None and xnum is not None:
        xint = np.abs(xmax - xmin) / float(xnum)
    elif xint is not None and xnum is not None:
        print 'WARNING: Both x interval and xnum were provided! Ignoring provided x interval...'
        xint = np.abs(xmax - xmin) / float(xnum)

    xnew = np.arange(xmin, xmax, xint)
    f = interpolate.interp1d(x, data, kind=mode)
    datanew = f(xnew)
    return xnew, datanew
