import os, sys, glob
import numpy as np
from tqdm import tqdm as tqdm

pdir = '/Volumes/labshared3/takumi'
datadirs = sorted(os.listdir(pdir))

# Search Stephane's M files

for i, dirname in tqdm(enumerate(datadirs)):
    datadir = os.path.join(pdir, dirname)

    print(datadir)
    for root, dirs, files in os.walk(datadir, topdown=False):
        # print(files)
        for name in files:
            path = os.path.join(root, name)
            # print(name, path[:-5] == '.hdf5')
            # if path[:-3] == '.h5' or path[:-5] == '.hdf5':
            if name.startswith('M') and (name.endswith('.hdf5') or name.endswith('.h5') ):
                print(path, 'YES')
       # for name in dirs:
       #     print(name)
       #    print(os.path.join(root, name))