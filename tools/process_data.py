"""
Module to process multidimensional numpy arrays (cleaning and interpolation)
Numpy Masking module is heavily implemented in this module.
- author: takumi
"""
import numpy as np
import copy
import numpy.ma as ma
from scipy import interpolate
from scipy.stats import binned_statistic

# Data cleaning (Masking / Filtering)

def interpolate_using_mask(arr, mask):
    """
    Conduct linear interpolation for data points where their mask values are True

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
    arr1 = copy.deepcopy(arr)
    arr2T = copy.deepcopy(arr).T

    f0 = np.flatnonzero(mask)
    f1 = np.flatnonzero(~mask)

    arr1[mask] = np.interp(f0, f1, arr1[~mask])

    f0 = np.flatnonzero(mask.T)
    f1 = np.flatnonzero(~mask.T)
    arr2T[mask.T] = np.interp(f0, f1, arr1.T[~(mask.T)])
    arr2 = arr2T.T

    arr = (arr1 + arr2) * 0.5
    return arr

def interpolate_using_mask_new(arr, mask, method='cubic'):
    """
    Developing...Too much computation time.
    Interpolates arr using a boolean mask
    ... This method flattens an array during interpolation. Therefore, the interpolated values at the edges should not be trusted.
    Parameters
    ----------
    arr: N-d array
    mask: N-d boolean array
    method: interpolation method of scipy griddata- linear, cubic

    Returns
    -------
    arr_interp: interpolated N-d array

    """
    dim = len(arr.shape)
    if dim < 3:
        x, y = np.arange(arr.shape[1]), np.arange(arr.shape[0])
        xx, yy = np.meshgrid(x, y)
        x1, y1, arr1 = xx[~mask], yy[~mask], arr[~mask]
        arr_interp = interpolate.griddata((x1, y1), arr1, (xx, yy), method=method)
    elif dim == 3:
        x, y, t = np.arange(arr.shape[1]), np.arange(arr.shape[0]),  np.arange(arr.shape[2])
        xx, yy = np.meshgrid(x, y)

        stack_m = lambda p: np.dstack((m, p))
        for i in range(arr.shape[2]-1):
            m = xx
            xx = stack_m(xx)
            m = yy
            yy = stack_m(yy)
        print xx.shape, yy.shape


        arr_interp = []
        for tt in range(arr.shape[2]):
            x1, y1, arr1 = xx[~mask], yy[~mask], arr[~mask]
            arr_interp_frac = interpolate.griddata((x1, y1), arr1, (xx, yy), method=method)
            arr_interp = np.stack((arr_interp, arr_interp_frac), axis=2)
    return arr_interp


def get_mask_for_unphysical(U, cutoffU=2000., fill_value=99999., verbose=True):
    """
    Returns a mask (N-dim boolean array). If elements were below/above a cutoff, np.nan, or np.inf, then they get masked.
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
    U = np.array(U)
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

def get_mask_for_unphysical2(U, low_tld=-2000., high_tld=2000., fill_value=99999., verbose=True):
    """
    Returns a mask (N-dim boolean array). If elements were below low_tld OR above high_tld, np.nan, or np.inf, then they get masked.
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
    U = np.array(U)
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
    b = ma.masked_greater(U_fixed, high_tld)
    c = ma.masked_less(U_fixed, low_tld)
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

def get_mask_for_unphysical3(U, low_tld=-2000., high_tld=2000., diff_tld=5, fill_value=99999., verbose=True):
    """
    Returns a mask (N-dim boolean array). If elements were below low_tld OR above high_tld, np.nan, or np.inf,
    or the absolute difference between successive values were above diff_tld, then they get masked.
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
    U = np.array(U)
    try:
        Udiff = np.diff(U)
        Udiff = np.concatenate((np.array([diff_tld*2]), Udiff))  # Add one element to have the same length as U, which will be deleted later
    except TypeError:
        Udiff = np.array([0])
    #print U, Udiff
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
        print '...number of masked elements by masked_invalid(nan and inf): ' + str(n_invalid)
    # Update the mask to False (no masking)
    U_fixed.mask = False



    # Mask unreasonable values of U_fixed
    mask_greater = ma.masked_greater(U_fixed, high_tld)
    mask_less = ma.masked_less(U_fixed, low_tld)
    mask_diff_greater = ma.masked_greater(Udiff, diff_tld)
    mask_diff_less = ma.masked_less(Udiff, -diff_tld)

    # Number of elements masked
    n_greater = ma.count_masked(mask_greater) - n_invalid
    n_less = ma.count_masked(mask_less)
    n_diff_greater = ma.count_masked(mask_diff_greater)
    n__diff_less = ma.count_masked(mask_diff_less)
    if verbose:
        print '...number of masked elements greater than cutoff: ' + str(n_greater)
        print '...number of masked elements less than -cutoff: ' + str(n_less)
        print '...number of masked elements greater than difference cutoff: ' + str(n_diff_greater)
        print '...number of masked elements less than -difference cutoff: ' + str(n__diff_less)

    # Generate a mask for all nonsense values in the array U
    mask = ~(~mask_greater.mask * ~mask_less.mask * ~mask_diff_greater.mask * ~mask_diff_less.mask)

    U_masked_unphysical = ma.array(U_fixed, mask=mask)
    n_total = ma.count_masked(U_masked_unphysical)
    # U_filled = ma.filled(d, fill_value)

    #Total number of elements in U
    N = 1
    for i in range(len(U.shape)):
        N *= U.shape[i]
    if verbose:
        print '...total number of unphysical values: ' + str(ma.count_masked(U_masked_unphysical)) + '  (' + str((float(n_total)/N*100)) + '%)\n'
    #print mask
    #print U_masked_unphysical
    return mask

def get_mask_for_nan_and_inf(U):
    """
    Returns a mask for nan and inf values in a multidimensional array U
    Parameters
    ----------
    U: N-d array

    Returns
    -------

    """
    U = np.array(U)
    U_masked_invalid = ma.masked_invalid(U)
    return U_masked_invalid.mask

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
    U_filled  numpy array

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
    elif mode=='greateinterpolate_using_maskrequal' or mode=='geq':
        U_masked = ma.masked_greater_equal(U, cutoff)
    return U_masked.mask


# Cleaning M class objects
def clean_vdata(M, cutoffU=2000, fill_value=np.nan, verbose=True):
    """
    Clean M class objects.
    Parameters
    ----------
    M
    cutoffU
    fill_value
    verbose

    Returns
    -------

    """
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

def delete_masked_elements(data, mask):
    """
    Deletes elements of data using mask, and returns a 1d array
    Parameters
    ----------
    data: N-d array
    mask: N-d array, bool

    Returns
    -------
    compressed_data

    """
    data_masked = ma.array(data, mask=mask)
    compressed_data = data_masked.compressed()
    '...Reduced data using a given mask'
    return compressed_data


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
    data1_masked = ma.array(data1, mask=mask)
    data2_masked = ma.array(data2, mask=mask)
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

def interpolate_1Darrays(x, data, xint=None, xnum=None, xmax=None, xmin=None, mode='cubic'):
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
    if xmax is None:
        xmax = np.max(x)
    if xmin is None:
        xmin = np.min(x)
    if xmax > np.max(x):
        x = np.concatenate([x, [xmax]])
        data = np.concatenate([data, [data[-1]]])
    if xmin < np.min(x):
        x = np.concatenate([[xmin], x])
        data = np.concatenate([ [data[0]], data])

    if xint is None and xnum is None:
        # Default is generate 10 times more data points
        xnum = len(x)*10
        xint = np.abs(xmax-xmin)/float(xnum)
    elif xint is None and xnum is not None:
        xint = np.abs(xmax - xmin) / float(xnum)
    elif xint is not None and xnum is not None:
        print 'WARNING: Both x interval and xnum were provided! Ignoring provided x interval...'
        xint = np.abs(xmax - xmin) / float(xnum)

    xnew = np.arange(xmin, xmax, xint)
    # check xnew has a length xnum
    if xnum is not None:
        if len(xnew) > xnum:
            excess = len(xnew) - xnum
            xnew = xnew[:-excess]
    f = interpolate.interp1d(x, data, kind=mode)
    datanew = f(xnew)
    return xnew, datanew


## Filter
def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y



# PDF
def pdf(data, nbins=10):
    """
    Returns a pdf of data (bins and hist with the same length)
    Parameters
    ----------
    data
    nbins

    Returns
    -------
    bins
    hist

    """
    data = np.asarray(data)

    if np.isnan(data).any():
        mask = get_mask_for_nan_and_inf(data)
        data = delete_masked_elements(data, mask) # returns a 1d array

    # Get a normalized histogram
    hist, bins = np.histogram(data.flatten(), bins=nbins, density=True)
    # len(bins) = len(hist) + 1
    # Get middle points for plotting sake.
    bins1 = np.roll(bins, 1)
    bins = (bins1 + bins) / 2.
    bins = np.delete(bins, 0)
    return bins, hist


## Binning data
def bin_data(arg, data, nbins=25):
    """
    Bins a pair of 1d data, and returns 1d data (binned_arguments, binned_data, std).
    One can plot these using axes.errorbar(arguments, data, std) or errorfill(arguments, data, std).
    Link: https://mycourses.aalto.fi/pluginfile.php/146910/mod_resource/content/1/binning_tutorial.pdf
    Parameters
    ----------
    arg
    data
    nbins

    Returns
    -------
    bin_centers: an array of centers of bins
    bin_averages: an array of magnitudes (average of data points in each bin)
    bin_stdevs: an array of stds for each bin

    """
    bin_centers, _, _ = binned_statistic(arg, arg, statistic='mean', bins=nbins)
    bin_averages, _, _ = binned_statistic(arg, data, statistic='mean', bins=nbins)
    bin_stdevs, _, _ = binned_statistic(arg, data, statistic='std', bins=nbins)
    return bin_centers, bin_averages, bin_stdevs







## get velocity from position, time arrays
def compute_velocity_simple(time, pos):
    """
    Compute velocity given that position and time arrays are provided
    - Use np.gradient should be enough for most of the purposes, but this method is much simpler, and more versatile
    - This does not care if time array is not evenly spaced.

    Parameters
    ----------
    pos : 1d array with length N
    time : 1d array with length N

    Returns
    -------
    velocity : 1d array with length N-1
    time_new : 1d array with length N-1
    """
    pos, time = np.array(pos), np.array(time)
    delta_pos = np.ediff1d(pos)
    delta_time = np.ediff1d(time)

    time_new = (time[1:] + time[:-1]) / 2.

    velocity = delta_pos / delta_time
    return time_new, velocity
