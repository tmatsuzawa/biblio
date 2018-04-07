import numpy as np


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
    arr1, arr2 = zip(*sorted(zip(arr1,arr2)))
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
    print indices, indices.shape
    if indices.shape==(1,0):
        print 'No sign flip in the array! Returning [0]...'
        return [0]

    if indices[0][0] == 0:
        # Detecting the first element is often a false alarm. Default is to delete the first element from the indices.
        if delete_first_index:
            indices = np.delete(indices, 0)
    return np.array(indices).flatten()
