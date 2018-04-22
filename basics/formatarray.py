import numpy as np
from scipy import ndimage

# Array Formatting
def array2chunks(l, chunksize):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), chunksize):
        yield l[i:i + chunksize]

def array2nchunks(l, n):
    """Yield n successive chunks from l."""
    chunksize = int(round(len(l) / n))
    for i in xrange(0, len(l), chunksize):
        yield l[i:i + chunksize]


def extend_1darray_fill(arr, newarrsize, fill_value=np.nan):
    """"""
    arr = np.array(arr)
    if len(arr) < newarrsize:
        return np.pad(arr, (0, newarrsize - len(arr)), 'constant', constant_values=(np.nan, np.nan))
    else:
        print 'Original array is bigger than new array. Returning the original array...'
        return arr

def find_nearest(array, value):
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
    return np.argmin(array), np.amin(array)

def find_max(array):
    """
    Find where maximum value of array is
    Parameters
    ----------
    array

    Returns
    -------

    """
    return np.argmax(array), np.amax(array)

# Array Manipulation
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
    indices : list   +1 if there is a sign flip. Otherwise, 0.  e.g. [1 0 1 0 0 1 1] (if zero_first_element==True)

"""
    arr = np.array(arr)
    arrsign = np.sign(arr)
    signchange = ((np.roll(arrsign, 1) - arrsign) != 0).astype(int)
    indices = np.array(np.where(signchange == 1))
    # Print indices, indices.shape
    if indices.shape==(1,0):
        print 'No sign flip in the array! Returning [0]...'
        return [0]

    if indices[0][0] == 0:
        # Detecting the first element is often a false alarm. Default is to delete the first element from the indices.
        if delete_first_index:
            indices = np.delete(indices, 0)
    return np.array(indices).flatten()


def get_values_from_multidim_array_at_coord(data_arr, x, y, order=3):
    """
    Returns values at spcific coordinates (indices) even if the coordinates are expressed as decimal numbers
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
    if not type(x) == 'list' or  type(x) == 'numpy.ndarray':
        x = [x]
        y = [y]
    # make sure all arrays are numpy arrays
    x = np.array(x)
    y = np.array(y)
    data_arr = np.array(data_arr)

    coord = [x, y]

    values = ndimage.map_coordinates(data_arr, coord, order=order)
    return values


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
    arr

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