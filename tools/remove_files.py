


import os
import numpy as np
import glob
import os.path

def remove(dir, ncycle = 40, dutycycle=0.85):
    n = int(ncycle * dutycycle)

    filelist = glob.glob(dir + '/*.tiff')
    for file in filelist:
        start = file.find('im') + 2
        end = file.find('.tiff', start)
        filenum = int(file[start:end])

        if filenum % ncycle == 0 or filenum % ncycle > n:
            os.remove(file)
    print 'Done'


    return filelist