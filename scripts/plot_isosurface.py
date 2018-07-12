"""
Plots isosurfaces of time-averaged energy distribution of turbulent blob
... Assumes that there are text files that store data like (x,y,E)
author takumi
"""
import argparse
import sys
import numpy as np
from scipy import ndimage, interpolate

sys.path.append('/Users/stephane/Documents/git/takumi/')
import library.tools.process_data as process
import library.tools.handle_data as dhandle
import library.display.graph as graph
import library.manager.file_architecture as file_architecture

# new grid spacing in mm
__xint__, __yint__, __zint__ = 1., 1., 1.


def interp_flow_component(x, y, z, data, xint=__xint__, yint=__yint__, zint=__zint__, method='linear'):
    points = [(xx, yy, zz) for xx, yy, zz in zip(x, y, z)]
    # grid_ind0, grid_ind1 = mgrid[0:imsize[0], 0:imsize[1]]
    # griddata = interpolate.griddata(uxindicies, ux, (grid_ind0, grid_ind1), method=method, fill_value=0.)
    xmin, xmax, ymin, ymax, zmin, zmax = np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z)
    xnew, ynew, znew = np.arange(xmin, xmax + xint, xint), np.arange(ymin, ymax + yint, yint), np.arange(zmin, zmax + zint, zint)
    Xnew, Ynew, Znew = np.meshgrid(xnew, ynew, znew)
    griddata = interpolate.griddata(points, data, (Xnew, Ynew, Znew), method=method, fill_value=np.nan)

    return Xnew, Ynew, griddata


if __name__ == '__main__':
    """
    Pass a absolute path of the cine files through '-input' e.g. -input '/absolute/path/to/cineDir/*.cine'
    """
    parser = argparse.ArgumentParser(description='')
    # Locate Data
    ## Method 1: date
    parser.add_argument('-d', metavar='date', type=str, default=None,
                        help='Date to be processed. Python will look for it in the folders specified in file_architecture.py')
    ## Method 2: Abs. path to data directory
    parser.add_argument('-datadir', dest='datadir', default=None, type=str,
                        help='Absolute path to a directory where slice# folders live')
    # Settings of Multi-layer PIV
    parser.add_argument('-nslices', dest='nslices', type=int, default=17,
                        help='Number of PIV slices')
    # Physical orientation
    parser.add_argument('-dist', dest='dist', type=float, default=None,
                        help='Distance between laser and center of a box')
    parser.add_argument('-angle', dest='theta', type=float, default=None,
                        help='sweeping angle / 2')
    parser.add_argument('-boxwidth', dest='boxwidth', type=float, default=254,
                        help='')

    parser.add_argument('-cineindex', dest='cineindex', type=int, default=1,
                        help='Index associated to cine')

    # PARSE COMMAND LINE INPUTS
    args = parser.parse_args()
    date = args.date
    datadir = args.datadir
    nslices = args.nslices
    cineindex = args.cineindex
    dist = args.dist
    theta = args.theta
    boxwidth = args.boxwidth

    Z = np.linspace(0, 2 * (dist+boxwidth/2.)*np.tan(theta), __zint__)



    subdir = '/AnalysisResults/Time_averaged_Plots_' + cineindex + '/'
    for slicenum in range(nslices):
        if datadir == None:
            datadir = file_architecture.get_dir(date)
        datapath = datadir + 'slice' + str(slicenum) +'/timeaveragedE_slice' + str(slicenum) + '.txt'
        key, data, counter = dhandle.generate_data_dct(datapath, separation=' ')






