# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:10:53 2015
Created on Wed Mar 18 16:40:47 2015
Created on Thu Mar 12 19:00:47 2015
@author: stephane
"""

import numpy as np
import os.path
import turbulence.tools.browse as browse
import turbulence.tools.pickle_m as pickle_m
from turbulence.mdata.Mdata import Mdata

"""
Created on Thu Mar 12 14:42:10 2015
@author: stephane
Old class to process PIV data from pyPIV algorithm (from Ivo Peters)
The use of static methods make impossible to reload old pickle files if the class has been edited since
"""


class Mdata_pyPIV(Mdata):
    def __init__(self, S, dataDir, mindex, ref=0):
        self.fieldnames = {'x': 'x', 'y': 'y', 'ux': 'ux', 'uy': 'uy'}
        self.frmt = 'npz'
        self.rootfile = 'PIVData'  # to be implemented !

        super(Mdata_pyPIV, self).__init__(S, dataDir, mindex, ref)

    def get_fileList(self, display=False):
        fileList, n = browse.get_fileList(self.dataDir + '/', self.frmt, display)
        if n == 0:
            print("No PIV files found ! at " + self.dataDir)
        return fileList, n

    def load_ref(self):
        # do it locally, from the first image
        fileList, nt = self.get_fileList(True)
        self.im_index = [browse.get_number(name, self.rootfile, self.frmt, False) for name in fileList]

        if nt > 0:
            Ref = np.load(fileList[0])
        else:
            Ref = None
        #        S.close()
        return Ref

    def read_ref(self, Ref):
        return super(Mdata_pyPIV, self).read_ref(Ref)

    def set_dimensions(self):
        return super(Mdata_pyPIV, self).set_dimensions()

    def set_scales(self):
        # ft is in ms, time elapsed between two consecutive images
        ft = 1000. / self.param.fps
        # timescale is the time resolution of the PIV  measurements
        num = browse.get_number(self.dataDir, "_at", "fps")
        if num > 0:
            timescale = num * ft / 1000.
        else:
            timescale = 1

        fx = self.param.fx

        return ft, timescale, fx

    def shape(self):
        return self.nx, self.ny, self.nt
        # read nx and ny from the ref picture,
        # read nt from the number of associated files

    def time_axis(self):
        # generation of a time axis
        # from the known value of fps and the t0 instant, generate an axis of time
        # t0 represent the image number at t=0
        im0 = self.param.im0
        if im0 is not None:
            # print("Set origin of times")
            t0 = (self.im_index[0] - im0) * self.ft
        else:
            print("No time reference found, origin of time arbitrary chosen at the beginning")
            t0 = 0.
        # time axis in ms
        # suppose that the frame are regularly spaced (!)
        # define it from the m_index list ??
        t = np.arange(t0, t0 + self.nt * self.ft, self.ft)
        return t

    def space_axis(self):
        x = self.U_ref[self.fieldnames['x']]
        y = self.U_ref[self.fieldnames['y']]

        if self.param.x0 is None:
            print("No space reference found, origin of space arbitrary chosen at the up left corner")
            x0 = 0.
            y0 = 0.
        else:
            x0 = self.param.x0
            y0 = self.param.y0
        x = (x - x0) * self.fx
        y = -(y - y0) * self.fx
        # if Sdata.param.sideview=='bv'
        #    ...
        if not (self.param.angle == 0):
            print('rotation of the data needed !')
        return x, y

    def load_measure(self):
        self.load_measure_fromPyPIV()

    def load_measure_fromPyPIV(self):
        fileList, nt = self.get_fileList(True)

        self.nt = nt
        self.t = self.time_axis()

        if nt == 0:
            return False
        # update = True means no change if the files are already present in the file
        # from a directory where all the .npz are stored, load_PIVData generate the x,y,Ux,Uy fields of Sdata
        # use the Dir name to find the fps ??
        # scale from the PIV settings : to be upgraded to a PIVsettings object
        self.scale = 2.8 * 10 ** -5

        # save parameters here !
        print("Loading data ...")
        self.Ux = np.zeros(self.shape())
        self.Uy = np.zeros(self.shape())

        # additionnal processing on the velocity measurements for now due to wrong PIVsettings (fps ans xscale)
        fact_x = self.fx / self.scale
        fact_t = 1000 / self.ft
        fact = fact_x * fact_t

        for name in fileList:
            count = fileList.index(name)
            S = np.load(name)
            # print(fileList.index(name))
            if nt > 100:
                if count % (nt // 10) == 0:
                    pass
                #                    print(str(count*10//nt) + " %")

            self.Ux[:, :, fileList.index(name)] = fact * S[self.fieldnames['ux']]
            self.Uy[:, :, fileList.index(name)] = fact * S[self.fieldnames['uy']]

            S.close()

        # MaJ of the time axis

        print("Data loaded")
        return True

    def load(self, start=0, end=0, step=1):
        if end == 0:
            self.load_measure_fromPyPIV()
            return None

        fileList, nt = self.get_fileList(True)
        self.nt = nt

        # complete definition of the time axis
        self.t = self.time_axis()
        fact = self.scale_velocity()

        for i in range(start, end, step):
            name = fileList[i]
            Ux, Uy = self.load_field(name)
            self.Ux[:, :, i] = fact * Ux
            self.Uy[:, :, i] = fact * Uy

    def load_field(self, filename):
        S = np.load(filename)
        Ux = S[self.fieldnames['ux']]
        Uy = S[self.fieldnames['uy']]
        S.close()
        return Ux, Uy

    def scale_velocity(self):
        fact_x = self.fx / self.scale
        fact_t = 1 / self.ft

        return fact_x * fact_t

    def eraseData(self):
        self.Ux = np.zeros((0, 0, 0))
        self.Uy = np.zeros((0, 0, 0))

        # Write the Sdata in the indicate filename

    def write(self, data=False):
        ############# Depreciated
        print("Writing Measure in pickle file : only the PIV parameters are written")
        if not data:
            self.eraseData()
        # write the Sdata object in a .txt file using the package pickle
        # The filename is directly determined from the Identification number
        self.fileDir = self.Sdata.dirCine + "M_" + self.id.date + "/"
        if not os.path.isdir(self.fileDir):
            os.makedirs(self.fileDir)

        # default name for the file containing the data
        self.filename = self.fileDir + "M_" + self.id.get_id() + ".txt"
        pickle_m.write(self, self.filename)

        print("Data written")

    # Read the Sdata object from the indicate filename
    # static method, generate a Sdata from the loaded file
    @staticmethod
    def read(filename, data=False):
        m = pickle_m.read(filename)

        if data:
            m.load_measure()
        return m

    def get_index(self, x0, y0):
        # return the closest index compare to the given position of the grid
        # should depend on the orientation : if 90 angle, i and j roles are inverted

        if self.Sdata.param.angle == 90:
            ys = x0
            xs = y0
            x = self.y * 1000
            y = self.x * 1000

            labelx = 'x'
            labely = 'y'
        else:
            xs = x0
            ys = y0

            x = self.x * 1000
            y = self.y * 1000

            labelx = 'x'
            labely = 'y'

            # bounds in mm
        minx = np.min(x)
        maxx = np.max(x)

        miny = np.min(y)
        maxy = np.max(y)

        print('bounds:')
        print(labelx + ' in [' + str(minx) + ', ' + str(maxx) + '] mm')
        print(labely + ' in [' + str(miny) + ', ' + str(maxy) + '] mm')

        boolList = [x0 >= minx, x0 <= maxx, y0 >= minx, y0 <= maxx]
        if all(boolList):
            j = np.argmin(np.abs(x[0, :] - xs))
            i = np.argmin(np.abs(y[:, 0] - ys))
        else:
            print('position out of bound')
            i = -1
            j = -1

        return i, j

    @staticmethod
    def getloc(date, index, mindex):
        Dir = Mdata.getDir(date)
        filename = Dir + "M_" + date + "_" + str(index) + "_" + str(mindex) + ".txt"
        return filename

    @staticmethod
    def getDir(date):
        Dir = "/Volumes/labshared3/Stephane/Experiments/" + date + "/M_" + date + "/"
        return Dir

    @staticmethod
    def getloc_M(m):
        Dir = Mdata.getDir(m.id.date)
        filename = Dir + "M_" + m.id.date + "_" + str(m.id.index) + "_" + str(m.id.mindex) + ".txt"
        return filename
