# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:12:04 2015

@author: stephane
"""

import numpy as np
import turbulence.tools.browse as browse
import turbulence.tools.rw_data as rw_data
import turbulence.mdata.time_step_sample as time_step_sample
from turbulence.mdata.Mdata import Mdata

import os.path
import turbulence.manager.file_architecture as file_architecture


class Mdata_PIVlab(Mdata, object):
    """
    Class for Data obtained from PIVlab algorithm (?)
    """

    def __init__(self, generate=True, **kwargs):
        super(Mdata_PIVlab, self).__init__(generate=generate, **kwargs)

    def __name__(self):
        return 'Mdata_PIVlab'

    def gen(self, dataDir='', **kwargs):
        self.fieldnames = {'x': 'x', 'y': 'y', 'ux': 'u','uy': 'v'}  # Here are the attributes specific to a PIVlab measurement
        self.frmt = 'txt'
        self.im_indexA = []
        self.im_indexB = []
        self.get_PIVparams(dataDir)

        super(Mdata_PIVlab, self).gen(dataDir=dataDir, **kwargs)

    def load(self, data):
        super(Mdata_PIVlab, self).load(data)

    def load_all(self, **kwargs):
        super(Mdata_PIVlab, self).load_all(**kwargs)

    def add_param(self, name, unit):
        super(Mdata_PIVlab, self).add_param(name, unit)

    def get_fileList(self, display=False):
        return super(Mdata_PIVlab, self).get_fileList(False)

    def get_PIVparams(self, dataDir):
        self.Dt = browse.get_number(dataDir, '_Dt_', '_', display=False, from_end=False)
        self.W = browse.get_number(dataDir, '_W', 'pix', display=False, from_end=False)

    def load_ref(self):
        fileList, nt = self.get_fileList(False)
        self.ref_Header, Data1D = rw_data.read_dataFile(fileList[0], ',', ',')

        new_Data1D = self.switch_keys(Data1D)
        Data2D = self.rescale_data(new_Data1D)
        return Data2D

    def read_ref(self, Ref):
        self.ref_indexA = browse.get_number(self.ref_Header[1], 'A: im', '.tiff', from_end=False)  # image names are localized in the second line of the ASCII file
        self.ref_indexB = browse.get_number(self.ref_Header[1], 'B: im', '.tiff', from_end=False)
        return super(Mdata_PIVlab, self).read_ref(Ref)

    def switch_keys(self, data, display=False):
        keys = data.keys()
        new_data = {}
        for k in keys:
            if not 'vorticity' in k:
                new_key = k[0]
            else:
                new_key = 'vort'
            new_data[new_key] = data[k]
            if display:
                print(k)
        return new_data

    def set_dimension_init(self, Data):
        """
        Returns the size of the 2d array (x,y)
        """
        minx = min(Data[self.fieldnames['x']])
        miny = max(Data[self.fieldnames['y']])  # why does this become minimum y? - Takumi 08/15/17

        nx = Data[self.fieldnames['x']].count(minx)
        ny = Data[self.fieldnames['y']].count(miny)  # print((nx,ny))
        return nx, ny

    def set_dimensions(self):
        return super(Mdata_PIVlab, self).set_dimensions()

    def shape(self):
        return super(Mdata_PIVlab, self).shape()

    def set_scales(self):
        # ft is in ms, time elapsed between two consecutive images
        print(self.ref_indexB)
        print(self.ref_indexA)
        Dt = self.ref_indexB - self.ref_indexA

        if hasattr(self.param, 'fps'):
            print('Parameter fps :' + str(self.param.fps))
            ft = Dt * 1000. / self.param.fps  # -> no global time axis : better to define it image per image
            # this function is thus depreciated : however, ft and timescale can be defined from the first couple of images
            timescale = ft / 1000
            fx = self.param.fx
        else:
            ft = 1
            fx = 1
            timescale = 1
        return ft, timescale, fx

    def time_axis(self):
        # generation of a time axis : not for PIVlab, use directly the header of the files (much more precise and flexible !)
        fileList, nt = self.get_fileList(False)

        if hasattr(self.Sdata, 'fileCine'):
            if os.path.isfile(self.Sdata.fileCine):
                times = time_step_sample.get_cine_time(self.Sdata.fileCine, True)
            else:
                # do a global search using file_architecture.py
                Dircine = file_architecture.get_dir(self.Id.date)
                fileCine = Dircine + '/' + os.path.basename(self.Sdata.fileCine)
                print(fileCine)
                times = time_step_sample.get_cine_time(fileCine, True)
        else:
            times = None

            #   self.im_index=np.zeros(nt)
        self.im_indexA = np.zeros(nt)
        self.im_indexB = np.zeros(nt)

        c = 0
        for i, filename in enumerate(fileList):
            Header = rw_data.read_Header(filename, ',')

            if Header == []:
                c += 1
                print("Header missing for : " + filename + ". delete file")
                print(Header)
                os.remove(filename)
            else:
                indexA = browse.get_number(Header[1], 'A: im', '.tiff', from_end=False)  # name of the image is localized on the second line of the ASCII file
                indexB = browse.get_number(Header[1], 'B: im', '.tiff', from_end=False)
                if indexA == 0:  # Header is given in another format
                    indexA = browse.get_number(Header[1], ': im', '.tiff', from_end=False)  # name of the image is localized on the second line of the ASCII file
                    indexB = browse.get_number(Header[1], '& im', '.tiff', from_end=False)
                self.im_indexA[i] = indexA
                self.im_indexB[i] = indexB
            self.im_index = self.im_indexA.tolist()

        # print("files un-able to read : "+str(c))

        if times is not None:
            a=self.im_indexA.astype(int)  #Convert a numpy array into int for later use as a counter  -added by takumi 08/15/2017
            b=self.im_indexB.astype(int)  #Convert a numpy array into int for later use as a counter  -added by takumi 08/15/2017

            Dt = [times[i] - times[j] for i, j in zip(b, a)]
            t = [times[int(i)] for i in self.im_index]
        else:
            Dt = [i - j for i, j in zip(self.im_indexB, self.im_indexA)]
            t = [int(i) for i in self.im_index]

        # super(Mdata_PIVlab,self).time_axis()
        return t, Dt

    def space_axis(self):
        return super(Mdata_PIVlab, self).space_axis()

    def rescale_data(self, Data1D):
        if not hasattr(self, 'nx'):
            self.nx, self.ny = self.set_dimension_init(
                Data1D)  # Data is given in txt file, so in 1 column, reshape it into 2D numpy array
        if not (self.nx * self.ny == len(Data1D['x'])):
            print("Data does not correspond to a rectangular matrix !")
            Data2D = None
        else:
            Data2D = {self.fieldnames[key]: np.zeros((self.nx, self.ny)) for key in self.fieldnames}
            for key in self.fieldnames:
                field = self.fieldnames[key]
                Data_extract = np.reshape(np.array(Data1D[field]),
                                          (self.ny, self.nx))  # transpose the Data (!) to fit with the other PIV soft
                Data2D[field] = np.transpose(
                    Data_extract)  # y varies first, then x : the matrix thus has to be transposed
        return Data2D

    def load_measure(self, rescale=True):
        self.fieldnames = {'x': 'x', 'y': 'y', 'ux': 'u',
                           'uy': 'v'}  # Here is the attributes specific to a PIVlab measurement
        self.frmt = 'txt'

        self.im_indexA = []
        self.im_indexB = []

        self.get_PIVparams(self.dataDir)

        fileList, nt = self.get_fileList(True)
        self.nt = nt

        if nt == 0:
            return False

        # print("Set time axis")        #MaJ of the time axis
        self.t, self.Dt = self.time_axis()

        # print("Loading data ...")
        self.Ux = np.zeros(self.shape())
        self.Uy = np.zeros(self.shape())

        Dx = self.fx

        for i, name in enumerate(fileList):
            Header, Data = rw_data.read_dataFile(name, ',', ',')
            # print(Header)
            # Not useful anymore : the time axis and Dt serie is generated once at the beginning
            #   indexA=browse.get_number(Header[1],'A: im','.tiff') #name of the image is localized on the second line of the ASCII file
            #   indexB=browse.get_number(Header[1],'B: im','.tiff')
            # from the name of the image, get the time at which the frame was taken !!!
            ##### old _method (linear movie)            
            # Dt=self.ft*(indexB-indexA)
            #    print(times)
            ###### from the time recorded in the cine file
            # Dt=times[indexB]-times[indexA]
            Dt = self.Dt[i]

            # print(self.im_indexA[i])
            new_Data = self.switch_keys(Data)  # use of standard keys

            Data2D = self.rescale_data(new_Data)

            if Data2D is not None:
                if rescale:
                    self.Ux[:, :, i] = Dx * Data2D['u'] / Dt
                    self.Uy[:, :, i] = Dx * Data2D['v'] / Dt
                else:
                    self.Ux[:, :, i] = Data2D['u']
                    self.Uy[:, :, i] = Data2D['v']
            else:
                print('Frame ' + str(i) + 'cannot be read')
            if nt > 100:
                if i % (nt // 100) == 0:
                    pass
                    # print('Dt = '+str(Dt*1000)+' ms, '+str(i*100//nt) + " %")
        # print("Data loaded")
        return True

    def eraseData(self):
        return super(Mdata_PIVlab, self).eraseData()

    # Write the Sdata in the indicate filename
    def write(self, data=False, **kwargs):
        super(Mdata_PIVlab, self).write(data, **kwargs)

    def write_hdf5(self):
        super(Mdata_PIVlab, self).write_hdf5()

        #  def get_index(self,x0,y0):
        #      return super(Mdata_PIVlab).get_index(x0,y0)

    ########################## Measurements #############################

    def get(self, field, **kwargs):
        super(Mdata_PIVlab, self).get(field, **kwargs)

    def get_cut(self):
        print("crazy")

    def compute(self, field, **kwargs):
        super(Mdata_PIVlab, self).compute(field, **kwargs)

    def measure(self, name, function, *args, **kwargs):
        super(Mdata_PIVlab, self).measure(name, function, *args, **kwargs)


######################### Plots ############################

def switch_keys(data, display=False):
    keys = data.keys()
    new_data = {}
    for k in keys:
        if not 'vorticity' in k:
            new_key = k[0]
        else:
            new_key = 'vort'
        new_data[new_key] = data[k]
        if display:
            print(k)
    return new_data
