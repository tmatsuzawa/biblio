import h5py
import numpy as np
import glob
import turbulence.tools.rw_data as rw_data
import turbulence.tools.browse as browse
import turbulence.jhtd.cutout as cutout
import datetime

'''
'''


def read(file, key_type=u'u'):
    """
    Read a JHDT file (h5py format) 
        read a h5py file
        extract from that the dataset names u'uX' where X denotes the time of the frame

    INPUT
    -----	
    file : string
        filename of the h5py file

    OUTPUT
    ------
    return a dictionnary with same keys as the initial set, each field containing one time step
    """
    f = h5py.File(file, 'r')
    # print(f.keys())
    data_keys = [key for key in f.keys() if key.find(key_type) >= 0]
    data = {}
    for key in data_keys:
        data[key] = f[key]
    return data


def read_chunk(folder, date):
    """
    Read all the data contained in the subfolder of the called folder
    """
    #    folder = '/Users/stephane/Documents/JHT_Database/Data/Spatial_measurement_1d_2016_03_30/Data'
    l = glob.glob(folder + '/zl*')
    print(folder)
    data = {}
    param = {}
    files_fail = []
    for i in range(len(l)):
        # print(i)
        files = glob.glob(l[i] + '/*.hdf5')
        for file in files:
            try:
                data_part = read(file)
                param[data_part.keys()[0]] = get_parameters(file)
                data.update(data_part)

            except:
                print(str(i) + ', skipped')
                #   print(file)
                files_fail.append(file)

    print('Number of failed loading : ' + str(len(files_fail)))
    # print(files_fail)
    # log_file = generate_log(files_fail,date)
    # print(log_file)

    #    Header,dset_list = rw_data.read_dataFile(log_file,Hdelimiter='\t',Ddelimiter='\t',Oneline=False)

    # cutout.recover(log_file)
    return data, param


def generate_log(files_fail, date=None, rootdir='/Users/npmitchell/Dropbox/Soft_Matter/turbulence/jhtd/data/'):
    """

    Parameters
    ----------
    files_fail
    date

    Returns
    -------

    """
    if date is None:
        date = datetime.date

    dirbase = rootdir + 'spatial_measurement_2d_' + date + '/'
    List_key = ['zl', 'yl', 'xl', 't0', 'tl', 'y0', 'x0', 'z0']

    log_file = dirbase + 'log.txt';
    f_log = open(log_file, 'w')
    rw_data.write_header(f_log, List_key)  # write the header

    for file in files_fail:
        #    print(file)
        param = get_parameters(file)
        rw_data.write_line(f_log, param)
    f_log.close()
    print(log_file + ' generated')

    return log_file


def get_parameters(file):
    """

    Parameters
    ----------
    file

    Returns
    -------

    """
    List_key = ['zl', 'yl', 'xl', 't0', 'tl', 'y0', 'x0', 'z0']

    param = {}
    for k in List_key[:-1]:
        param[k] = int(browse.get_string(file, '_' + k + '_', end='_', display=False))
    k = List_key[-1]
    param[k] = int(browse.get_string(file, '_' + k + '_', end='/', display=False))

    return param


def vlist(data, rm_nan=True):
    """
    Extract the velocity components from a JHTD data format. Spatial coherence is lost (return a list of single data
    points)

    Parameters
    ----------
    data : JHTD data format

    Returns
    -------
    U : numpy array of 1d1c velocity components
        each element corresponds to one point 3C of velocity
    """
    U = []
    for key in data.keys():
        # wrap the three components in one 3d matrix (by arbitrary multiplying the first dimension by 3 : spatial
        # information is lost)
        dim = data[key].shape
        dim = (dim[0] * 3, dim[1], dim[2], 1)
        Upart = np.reshape(data[key], dim)

        # wrap to 1d vector every single component
        dimensions = Upart.shape
        N = np.product(np.asarray(dimensions))
        U_1d = np.reshape(Upart, (N,))
        if rm_nan:
            if np.isnan(U_1d).any():
                print('Nan value encountered : removed from the data')
            U_1d = U_1d[~np.isnan(U_1d)]

        U += np.ndarray.tolist(U_1d)

    return U


def generate_struct(data, param):
    """

    Parameters
    ----------
    data
    param

    Returns
    -------
    jhtd_data
    """
    jhtd_data = JHTDdata(data, param)
    return jhtd_data
    # from a dataset, generate a Mdata structure
