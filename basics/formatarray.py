import numpy as np
from scipy import ndimage
import copy
import library.tools.process_data as process

# Basics
def find_nearest(array, value, option='normal'):
    """
    Find an element and its index closest to 'value' in 'array'
    Parameters
    ----------
    array
    value

    Returns
    -------
    idx: index of the array where the closest value to 'value' is stored in 'array'
    array[idx]: value closest to 'value' in 'array'

    """
    # get the nearest value such that the element in the array is LESS than the specified 'value'
    if option == 'less':
        array_new = copy.copy(array)
        array_new[array_new > value] = np.nan
        idx = np.nanargmin(np.abs(array_new - value))
        return idx, array_new[idx]
    # get the nearest value such that the element in the array is GREATER than the specified 'value'
    if option == 'greater':
        array_new = copy.copy(array)
        array_new[array_new < value] = np.nan
        idx = np.nanargmin(np.abs(array_new - value))
        return idx, array_new[idx]
    else:
        idx = (np.abs(array-value)).argmin()
        return idx, array[idx]

def find_min(array):
    """
    Find where minimum value of array is
    Parameters
    ----------
    array

    Returns
    -------

    """
    # convert it to arrays if list is given
    if type(array) == list:
        array = np.array(array)
    args, minvalue = np.argmin(array), np.amin(array)
    args = np.unravel_index(args, array.shape)
    if len(args) == 1:
        return args[0], np.amin(array)
    else:
        return args, np.amin(array)


def find_max(array):
    """
    Find where maximum value of array is
    Parameters
    ----------
    array: N-d array

    Returns
    -------
    If list or 1d array is given, returns index as an integer;  args[0], np.amax(array)
    Otherwise, gives back a tuple and the maximum value of the array; args, np.amax(array)

    args: tuple or integer
    np.amax(array)
    """
    # convert it to arrays if list is given
    if type(array) == list:
        array = np.array(array)
    args, maxvalue = np.argmax(array), np.amax(array)
    args = np.unravel_index(args, array.shape)
    if len(args) == 1:
        return args[0], np.amax(array)
    else:
        return args, np.amax(array)

def find_centroid(array):
    """

    Parameters
    ----------
    array: 2d array

    Returns
    -------
    indices, array

    """
    return ndimage.measurements.center_of_mass(array)

def count_occurrences(arr, display=True):
    """
    Returns occurrances of items in an array in a dictionary

    Parameters
    ----------
    arr
    display: bool, If True, it prints occurrences

    Returns
    -------
    occur_dict : dictionary

    """
    unique, counts = np.unique(arr, return_counts=True)
    occur_dict = dict(zip(unique, counts))
    if display:
        print occur_dict
    return occur_dict



# Array sorting
def sort_two_arrays_using_order_of_first_array(arr1, arr2):
    """
    Sort arr1 and arr2 using the order of arr1
    e.g. a=[2,1,3], b=[4,1,9]-> a[1,2,3], b=[1,4,9]
    Parameters
    ----------
    arr1
    arr2

    Returns
    -------

    """
    arr1, arr2 = zip(*sorted(zip(arr1, arr2)))
    return arr1, arr2


# Application
def detect_sign_flip(arr, delete_first_index=True):
    """
    Returns indices of an 1D array where its elements flip the sign
    e.g.  arr=[1,1,-1,-2,-3,4,-1] -> signchange=[1, 0, 1, 0, 0, 1, 1]
        -> indices=[0, 2, 5, 6] (if delete_first_index=False) or indices=[2, 5, 6] (delete_first_index=True)
    Parameters
    ----------
    arr : list or 1D numpy array e.g. [1,1,-1,-2,-3,4,-1]

    Returns
    -------
    indices : 1d array   +1 if there is a sign flip. Otherwise, 0.  e.g. [1 0 1 0 0 1 1] (if zero_first_element==True)

"""
    arr = np.array(arr)
    arrsign = np.sign(arr)
    signchange = ((np.roll(arrsign, 1) - arrsign) != 0).astype(int)
    indices = np.array(np.where(signchange == 1))
    # Print indices, indices.shape
    if indices.shape==(1, 0):
        print 'No sign flip in the array! Returning [0]...'
        return np.array([0])

    if indices[0][0] == 0:
        # Detecting the first element is often a false alarm. Default is to delete the first element from the indices.
        if delete_first_index:
            indices = np.delete(indices, 0)
    return np.array(indices).flatten()

def get_average_data_from_periodic_data(time, periodic_data, freq=1., interpolate_no=10, returnChunks=False):
    """
    get average data from periodic data
    i.e. the periodic data contains 10 periods, this will return data which is a period long, averaged over periods
    Parameters
    ----------
    periodic_data
    time: array,
    freq: float
    interpolate_no: number of interpolated points per data point
    returnChunks: bool If true, it returns averaged arrays and all chunks generated to produce the averaged data

    Returns
    -------
    time_short, data_mean, data_std: arrays of averaged data. These three arrays have the same length.

    Optional:
    time_chunks, data_chunks, time_chunks_int, data_chunks_int: lists of data separated into multiple chunks.
                                                                _int refers to interpolated chunks
    """
    data_chunk_2d, time_chunks_int, data_chunks_int = [], [], []
    # make sure that arrays are numpy arrays
    periodic_data, time = np.array(periodic_data), np.array(time)
    time = time - np.nanmin(time)

    # calculate period, total time, and number of cycles included in the data array
    period = 1. / freq
    total_time = np.max(time) - np.min(time)

    numcycles = int(np.ceil(total_time / period))

    time_chunks, data_chunks = [], []
    chunk_length = []
    for i in range(numcycles):
        tmin = i * period
        tmax = (i + 1) * period

        idx_max, tmax = find_nearest(time, tmax, option='less')
        idx_min, tmin = find_nearest(time, tmin, option='greater')
        time_chunks.append(time[idx_min: idx_max])
        data_chunks.append(periodic_data[idx_min: idx_max])
        chunk_length.append(idx_max - idx_min)

    # interpolate data if the length of the chunk is more than a half of the longest chunk
    # otherwise, throw it away
    # throuw away the last chunk as well
    indices_to_be_deleted = []
    for i in range(numcycles):
        if len(data_chunks[i]) < max(chunk_length) / 2 or (i == numcycles-1 and numcycles > 1):
            indices_to_be_deleted.append(i)
            continue
        else:
            time_chunks[i] = time_chunks[i] - np.min(time_chunks[i])
            time_chunk_int, data_chunk_int = process.interpolate_1Darrays(time_chunks[i], data_chunks[i],
                                                                          xnum=max(chunk_length)*interpolate_no, xmin=0, xmax=period, mode='linear')
            time_chunks_int.append(time_chunk_int)
            data_chunks_int.append(data_chunk_int)
    # delete chunks which did not have more than a half of the longest chunk
    data_chunks = [data_chunks[i] for i in range(numcycles) if i not in indices_to_be_deleted]
    numcycles = numcycles - len(indices_to_be_deleted)

    # make data_chunk_2d (which is currently 1D) into a 2D array
    data_chunk_2d = np.concatenate(np.transpose(data_chunks_int)).ravel().reshape(max(chunk_length)*interpolate_no,
                                                                                numcycles)  # <- Now, this is 2d array.
    time_short = time_chunks_int[0]

    # Calculate average and std
    data_mean = np.nanmean(data_chunk_2d, axis=1)
    data_std = np.nanstd(data_chunk_2d, axis=1)

    if returnChunks:
        return time_short, data_mean, data_std, time_chunks, data_chunks, time_chunks_int, data_chunks_int
    else:
        return time_short, data_mean, data_std



# Interpolation / map_coordinates etc.
def get_values_from_multidim_array_at_coord(data_arr, x, y, order=3):
    """
    Returns values at specific coordinates (indices) even if the coordinates are expressed as decimal numbers
    e.g.- a is a 2d array, and you would like to get a value at (x1, y1) = (1.2, 6.5).
          This method returns an interpolated value.
    Give coordinates (x1,y1), (x2, y2),... like [x1, x2, ...], [y1, y2, ...]
    Parameters
    ----------
    data_arr multi-dim array
    x
    y

    Returns
    -------
    value

    """
    if not type(x) == 'list' or type(x) == 'numpy.ndarray':
        x = [x]
        y = [y]
    # make sure all arrays are numpy arrays
    x = np.array(x)
    y = np.array(y)
    data_arr = np.array(data_arr)

    coord = [x, y]
    values = ndimage.map_coordinates(data_arr, coord, order=order)

    return values

def extend_1darray_fill(arr, newarrsize, fill_value=np.nan):
    """"""
    arr = np.array(arr)
    if len(arr) < newarrsize:
        return np.pad(arr, (0, newarrsize - len(arr)), 'constant', constant_values=(np.nan, np.nan))
    else:
        print 'Original array is bigger than new array. Returning the original array...'
        return arr



# Array Formatting
##1D
# Remove a certain portion of an array
def remove_first_n_perc_of_array(arr, percent=0.3):
    # make it into an array just in case
    arr = np.array(arr)
    return arr[int(len(arr)*percent):]

def remove_last_n_perc_of_array(arr, percent=0.3):
    # make it into an array just in case
    arr = np.array(arr)
    return arr[:int(len(arr)*(1.-percent))]


# Make chunks from a 1D array
def array2chunks(l, chunksize):
    """
    Yield successive n-sized chunks from l.
    ... 'yield' returns generators
    """
    for i in xrange(0, len(l), chunksize):
        yield l[i:i + chunksize]

def array2nchunks(l, n):
    """Yield n successive chunks from l."""
    chunksize = int(round(len(l) / n))
    for i in xrange(0, len(l), chunksize):
        yield l[i:i + chunksize]

##2D
# Make blocks from 2d arrays
def make_blocks_from_2d_array(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where n * nrows * ncols = arr.size
    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.

    Parameters
    ----------
    arr: M x N list or numpy array
    nrows:
    ncols

    Returns
    -------
    blocks: numpy array with shape (n, nrows, ncols)
    """

    arr = np.array(arr)
    h, w = arr.shape
    blocks = (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
    return blocks

#Divide a 2D array into four quadrants
def divide_2d_array_into_four_domains(arr, rx=0.5, ry=0.5):
    """
    Divide m x n matrix into four domains

    ################################
    #        -> x
    #   |           rx        1-rx
    # y v       <---------><--------->
    #        ^ | Domain 1 | Domain 3 |
    #     ry | |          |          |
    #        v  _____________________
    #        ^ |          |          |
    #  1-ry  | |          |          |
    #        v | Domain 2 | Domain 4 |
    ################################

    Parameters
    ----------
    arr: 2d array
    rx : float [0,1]
    ry : float [0,1]

    Returns
    -------

    """
    arr = np.array(arr)
    m, n = arr.shape  # NOTE THAT SHAPE RETURNS (NO OF ROWS * NO OF COLUMNS)
    mm, nn = int(round(m * ry)), int(round(n * rx))
    arr1, arr2 = arr[:mm, :nn], arr[mm:, :nn]
    arr3, arr4 = arr[:mm, nn:], arr[mm:, nn:]
    blocks = [arr1, arr2, arr3, arr4]
    return blocks

#Extract a small region (nx x ny) of arrays around a specified coordinate
def get_small_grids_around_coord(datagrid, xgrid, ygrid, x, y, nx, ny):
    """

    gives back a nx x ny matrix around (x, y) from xgrid, ygrid, datagrid


    ################################
    #        -> x
    #   |              2nx+1
    # y v            <-------->
    #           ____________________
    #          |      ________      |
    #  2ny+1 ^ |     |        |     |
    #        | |     |   x    |     |
    #        v |     |________|     |
    #          |____________________|
    #
    ################################

    Parameters
    ----------
    griddata
    xgrid
    ygrid
    x
    y
    nx
    ny

    Returns
    -------

    """

    def get_proper_indices_for_x(a, ncolumns):
        if a < 0:
            return 0
        elif a >= ncolumns:
            return int(ncolumns - 1)
        else:
            return int(a)

    def get_proper_indices_for_y(a, nrows):
        if a < 0:
            return 0
        elif a >= nrows:
            return int(nrows - 1)
        else:
            return int(a)
    nrows, ncolumns = datagrid.shape
    datagrid_around_coord = datagrid[get_proper_indices_for_y(y - ny, nrows): get_proper_indices_for_y(y + ny, nrows),
                            get_proper_indices_for_x(x - nx, ncolumns): get_proper_indices_for_x(x + nx, ncolumns)]
    xgrid_around_coord = xgrid[get_proper_indices_for_y(y - ny, nrows): get_proper_indices_for_y(y + ny, nrows),
                         get_proper_indices_for_x(x - nx, ncolumns): get_proper_indices_for_x(x + nx, ncolumns)]
    ygrid_around_coord = ygrid[get_proper_indices_for_y(y - ny, nrows): get_proper_indices_for_y(y + ny, nrows),
                         get_proper_indices_for_x(x - nx, ncolumns): get_proper_indices_for_x(x + nx, ncolumns)]
    return xgrid_around_coord, ygrid_around_coord, datagrid_around_coord

# Sort arrays
def sort2arr(arr2, arr1):
    """
    Sorted by the order of arr1
    Parameters
    ----------
    arr2
    arr1

    Returns
    -------
    arr2_sorted, arr1_sorted

    """
    zipped = zip(arr2, arr1)
    zipped_sorted = sorted(zipped, key=lambda x: x[1])
    arr2_sorted, arr1_sorted = zip(*zipped_sorted)
    return arr2_sorted, arr1_sorted