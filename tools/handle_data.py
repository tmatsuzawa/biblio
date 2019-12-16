'''
Module for organizing data from txt files for various computations
'''
import numpy as np
import os
import matplotlib.pyplot as plt


#cwd = os.getcwd()  # get the path of the current working directory
#filelist = os.listdir(cwd)  # list files in the directory

def generate_data_dct(dataPath, separation='\t'):
    """
    Read data from txt files, and make a dictionary
    Parameters
    ----------
    dataPath : string
        absolute path of the data txt file

    Returns
    -------
    data : dictionary
        keys are named as var0, var1, ...var#. Corresponding values are lists.
    """
    f = open(dataPath, 'r')
    counter=1
    data={}
    for line in f:
        if counter==1:
            key = line.split(separation)
            for x in range(0,len(key)):
                data["var{0}".format(x)] = []  # initialize lists var0, var1,...
        else:
            val = line.split(separation)
            for i,x in enumerate(val):
                try:
                    val[i]=float(val[i])
                except ValueError:
                    pass
            #print val
            #print len(key)
            for x in range(0, len(key)):
                data["var{0}".format(x)].append(val[x])
            #print data

        counter = counter+1
    f.close()
    return key, data, counter

def generate_data_dct_masked(dataPath, threshold=0.0, separation='\t'):
    """
    Read data from txt files, and make a dictionary with masked numpy arrays
    Parameters
    ----------
    dataPath : string
        absolute path of the data txt file
    separation : string, default=' '
        specify how data values are separated in the data file. e.g.- separation=',' for csv files
    threshold : float, default = 0.0
        this method generates a dictionary with numpy masked arrays. The array values below threshold will be masked.
    Returns
    -------
    dataMasked : dictionary
        a dictionary with numpy masked arrays.
        keys are named as var0, var1, ...var#. Corresponding values are numpy masked lists.
    """
    dataMasked={}   #initialize a dictionary where numpy maseked arrays will be stred
    key, data, counter = generate_data_dct(dataPath, separation)

    for x in range(0,len(key)):
        dataMasked["var{0}".format(x)] = []  # initialize lists var0, var1,...
        npDataArray=np.ma.array(data["var{0}".format(x)]) #Use numpy array for masking
        npMaskedDataArray = np.ma.masked_where(npDataArray < threshold, npDataArray)
        dataMasked["var{0}".format(x)]= npMaskedDataArray  #Mask values below the threshold
    return key, dataMasked, counter


def generate_data_dct_cropped(dataPath, threshold=0.0, separation='\t'):
    """
    Read data from txt files, and make a dictionary with masked numpy arrays
    Parameters
    ----------
    dataPath : string
    absolute path of the data txt file
    separation : string, default=' '
        specify how data values are separated in the data file. e.g.- separation=',' for csv files
    threshold : float, default = 0.0
        this method generates a dictionary with numpy masked arrays. The array values below threshold will be masked.
    Returns
    -------
    dataCropped : dictionary
        a dictionary with numpy arrays.
        keys are named as var0, var1, ...var#. Corresponding values are numpy arrays which are cropped into arryas that contain values and no masks.
    """
    dataCropped = {}  # initialize a dictionary where numpy masked arrays will be stored
    key, dataMasked, counter = generate_data_dct_masked(dataPath, threshold, separation)
    for x in range(0, len(key)):
        dataCropped["var{0}".format(x)] = []  # initialize lists var0, var1,...
        checkerArray=np.zeros((len(dataMasked["var{0}".format(x)])),dtype=bool)
        #if np.array_equal(dataMasked["var{0}".format(x)].mask, checkerArray)==False:
        try:
            if dataMasked["var{0}".format(x)].mask==False:
                dataCropped["var{0}".format(x)] = dataMasked["var{0}".format(x)]  # If var# does not have a mask, no need to crop the data!
        except ValueError:
            for y in range(0, len(dataMasked["var{0}".format(x)])):
                if dataMasked["var{0}".format(x)].mask[y]==False:  # Crop data so that it contains only values and no masks
                    #np.append(dataCropped["var{0}".format(x)],dataMasked["var{0}".format(x)][y])
                    dataCropped["var{0}".format(x)].append(dataMasked["var{0}".format(x)][y])
                continue
    return key, dataCropped, counter

def generate_data_dct_plottable(dataPath, threshold=0.0, separation='\t', varIndexForX=0):
    """
    Read data from txt files, and make a dictionary with masked numpy arrays
    Parameters
    ----------
    dataPath : string
    absolute path of the data txt file
    separation : string, default=' '
        specify how data values are separated in the data file. e.g.- separation=',' for csv files
    threshold : float, default = 0.0
        this method generates a dictionary with numpy masked arrays. The array values below threshold will be masked.
    varIndexForX : integer, default = 0
        This is an index to specify which variable you consider as x (controlling variable).
        e.g. if you consider plotting var5 vs var3, then varIndexForX = 3.
    Returns
    -------
    dataCropped : dictionary
        a dictionary with numpy arrays.
        keys are named as var0, var1, ...var#. Corresponding values are numpy arrays which are cropped into arryas that contain values and no masks.
    """

    key, dataMasked, counter = generate_data_dct_masked(dataPath, threshold=0.0, separation='\t')

    dataMaskedForPlot={}
    # for x in range(0, 2*len(key)):
    #     dataMaskedForPlot["var{0}".format(x)] = np.ma.zeros(len(dataMasked["var0"]))  # initialize lists var0, var1,...

    for x in range(0,2*len(key),2):
        #dataMaskedForPlot["var{0}".format(x)], dataMaskedForPlot["var{0}".format(x+1)] = [], []  # initialize lists var0, var1,...

        #npMaskedPlottableDataArray1, npMaskedPlottableDataArray2 = generate_plottable_arrays(dataMasked["var{0}".format(varIndexForX)], dataMasked["var{0}".format(x/2)])
        #dataMaskedForPlot["var{0}".format(x)]=npMaskedPlottableDataArray1
        #dataMaskedForPlot["var{0}".format(x+1)] = npMaskedPlottableDataArray2

        dataMaskedForPlot["var{0}".format(x)],dataMaskedForPlot["var{0}".format(x+1)]=generate_plottable_arrays(dataMasked["var{0}".format(varIndexForX)], dataMasked["var{0}".format(x/2)])

        print('var' + str(x) + '-Masked')
        #print dataMaskedForPlot["var{0}".format(x)]
        print(dataMaskedForPlot["var{0}".format(x)].mask)
        print('var' + str(x+1) + '-Masked')
        #print dataMaskedForPlot["var{0}".format(x+1)]
        print(dataMaskedForPlot["var{0}".format(x+1)].mask)

        # if not x==0:
        #     print 'var' + str(x-2) + '-Masked'
        #     print dataMaskedForPlot["var{0}".format(x)].mask
        #     print 'var' + str(x-1) + '-Masked'
        #     print dataMaskedForPlot["var{0}".format(x + 1)].mask


#Cropping
    dataCroppedForPlot = {}  # initialize a dictionary where numpy masked arrays will be stored
    for x in range(0, len(dataMaskedForPlot)):
        dataCroppedForPlot["var{0}".format(x)] = []  # initialize lists var0, var1,...
        checkerArray = np.zeros((len(dataMaskedForPlot["var{0}".format(x)])), dtype=bool) #Used to identify which lists need to be cropped for plotting
        print(dataMaskedForPlot["var{0}".format(x)].mask)
        print('var' + str(x))
        print(dataMaskedForPlot["var{0}".format(x)].mask)
        if np.array_equal(dataMaskedForPlot["var{0}".format(x)].mask, checkerArray):
                dataCroppedForPlot["var{0}".format(x)] = dataMaskedForPlot["var{0}".format(x)]  # If var# does not have a mask, no need to crop the data!
                print('var' + str(x) + ' will NOT be cropped! (no mask)')
        if not np.array_equal(dataMaskedForPlot["var{0}".format(x)].mask, checkerArray):
            print('var' + str(x) + ' will be  cropped!')
            for y in range(0, len(dataMaskedForPlot["var{0}".format(x)])):
                if dataMaskedForPlot["var{0}".format(x)].mask[y] == False:  # Crop data so that it contains only values and no masks
                    dataCroppedForPlot["var{0}".format(x)].append(dataMaskedForPlot["var{0}".format(x)][y])
                    #print 'var' + str(x) + '[' + str(y) +'] was appended to ' +  'var' + str(x)
        print(dataCroppedForPlot["var{0}".format(x)])

    return key, dataCroppedForPlot


def generate_plottable_arrays(maskedArray1, maskedArray2):
    newMaskedArray1 = maskedArray1
    newMaskedArray2 = maskedArray2
    mask1 = maskedArray1.mask
    mask2 = maskedArray2.mask
    #mask3 = np.zeros(len(maskedArray1), dtype=bool)
    notMask1,notMask2,notMask3=[],[],[]
    try:
        if mask1==False:
            try:
                if mask2==False:
                    mask3=False
                    print('1')
            except ValueError:
                mask3=mask2
                print('2')
    except ValueError:
            try:
                if mask2==False:
                    mask3=mask1
                    print('3')
            except ValueError:
                print('4')
                for x in range(0, len(maskedArray1)):
                    notMask1.append(not mask1[x])
                    notMask2.append(not mask2[x])
                    notMask3.append(notMask1[x]*notMask2[x])
                mask3 = [not x for x in notMask3]
    # print mask1
    # print mask2
    # print mask3

    #Remask the masked arrays with the new mask, mask3

    newMaskedArray1.mask=mask3
    newMaskedArray2.mask=mask3

    return newMaskedArray1, newMaskedArray2





def main(dataPath, threshold=0.0, separation=' '):
    # Generate data arrays from a txt file
    key, dataRaw, counter = generate_data_dct(dataPath, separation='\t')

    # Exclude na values from the arrays by masking the generated data arrays
    # Convention: -1 in the txt file means "na".
    key, dataMasked, counter = generate_data_dct_masked(dataPath, threshold=0.0, separation='\t')

    # Crop the Masked Data arrays so that the Cropped Data arrays contain only floats.
    key, dataCropped, counter = generate_data_dct_cropped(dataPath, threshold=0.0, separation='\t')

    # Instead of just cropping, you may crop the Masked Data arrays in such a way that they become immediately plottable.

    return