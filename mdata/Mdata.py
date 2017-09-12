# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:40:47 2015

@author: stephane
"""

import numpy as np
import os.path
import time
import h5py
import dpath
import turbulence.tools.browse as browse
import turbulence.mdata
import turbulence.mdata.Id as Id
import turbulence.mdata.Sdata as Sdata
import turbulence.mdata.param as param
import turbulence.tools.pickle_m as pickle_m
import turbulence.hdf5.h5py_convert as to_hdf5
import turbulence.manager.file_architecture as file_architecture
import turbulence.analysis.vgradient as vgradient

'''Mdata is a class for handling measurements made on raw data.
'''


class Mdata(object):
    """description
    """

    def __init__(self, generate=True, **kwargs):
        """
        Class generator
        """
        # A measure is associated to :
        # a Sdata (parameters of a cine file)
        # a dataSet

        if generate:
            self.gen(**kwargs)
        else:
            self.param = param.param(self, generate=False)
            self.Sdata = Sdata.Sdata(generate=False)

            if kwargs is not None:
                if 'dataDir' in kwargs.keys():
                    self.dataDir = kwargs['dataDir']

                    #   self.U_ref={'ux':None,'uy':None,'x':None,'y':None}

    def __name__(self):
        return 'Mdata'

    def gen(self, S=None, dataDir='', mindex=0, ref=0):
        """
        Generate Mdata from : Sdata for the header, then
        .hdf5 file if exist
        .txt (pickle) otherwise
        generate it from the raw data otherwise
        """
        if S is not None:
            self.Sdata = Sdata.Sdata(generate=False)
            self.Sdata.gen(fileCine=S.fileCine, index=S.Id.index, mindex=S.Id.mindex)
            print('------------------------------------------------------------')
            print('------------------------------------------------------------')
            print('------------------------------------------------------------')

            print(type(self.Sdata))
        else:
            print('------------------------------------------------------------')
            print('Mdata not associated to a cine file !')
            print('------------------------------------------------------------')

        self.dataDir = dataDir
        # Load parameters
        self.param = self.Sdata.param

        self.Id = Id.Id(self.Sdata, index=self.Sdata.Id.index, mindex=mindex)

        print(self.Id.get_id())
        # Data file
        # Data are not stored by default in Sdata, the data load must be called explicitly
        self.filePIV = ''  # associate the PIV file parameter ?

        #       print(self.Sdata.fileDir)
        self.fileDir = os.path.dirname(self.Sdata.fileDir[:-12]) + '/' + 'M_' + self.Id.date + '/'
        #       print(self.fileDir)
        # print(browse.get_string(os.path.dirname(self.Sdata.fileDir),start='',end='/',from_end=True))
        # self.fileDir+"M_" + self.Id.get_id() + frmt

        # print(self.get_filename())

        print(self.get_filename(frmt='.hdf5'))
        if os.path.exists(self.get_filename(frmt='.hdf5')):
            self.load_all(source=self.get_filename(frmt='.hdf5'), mindex=mindex)
        elif os.path.exists(self.get_filename('.txt')):
            try:
                self.load_all(source=self.get_filename('.txt'))
            except:
                print("Pickle object cannot be read anymore")
                pass
        else:
            print("load parameters manually")
            self.U_ref = self.read_ref(self.load_ref())
            # U_ref has attribute x,y, Ux, Uy
            self.nx, self.ny, self.nt = self.set_dimensions()

            self.ft, self.timescale, self.fx = self.set_scales()
            #        self.x,self.y,self.t=set_scales()
            self.x, self.y = self.space_axis()
            self.t = self.time_axis()

            # write the Sdata object in a .txt file using the package pickle
            # The filename is directly determined from the Identification number
            self.fileDir = self.Sdata.dirCine + "M_" + self.Id.date + "/"
            # trouble with the Id of the M file : find another (more     accurate way of classification)
            self.fileDir = file_architecture.os_c(self.fileDir)

    def load(self, data):
        """
        Load a hdf5 file group to the attributes of a class instance
        """
        self = to_hdf5.load(self, data)

    def load_all(self, source=None, mindex=0):
        """
        Load the data from a hdf5 file
        """
        if source is None:
            if hasattr(self, 'filename'):
                filename = self.filename + '.hdf5'
        else:
            filename = source

        f = to_hdf5.open(filename)

        if 'Sdata' in f['Mdata'].keys():
            self.load(f['Mdata'])
        else:
            names = ['Sdata', 'param', 'Id']
            for name in names:
                self.load_rec(f, name)

        if self.Id == 'pointer':
            self.Id = self.Sdata.Id
            self.Id.mindex = mindex
        f.close()

        # print(self.Id)
        print(self.Ux.shape)

    def load_rec(self, f, name, key=''):
        key_fix = 'Mdata'
        #  print(key_fix+key)
        try:
            #   print("toto : " +str(dpath.util.search(dict(f[key_fix+key]),'*mdata.'+name+'*').keys()))
            keys = dpath.util.search(dict(f[key_fix + key]), '*mdata.' + name + '*').keys()
        except:
            print("Nothing to look for anymore")
            return None
        if not keys == []:
            # print(name)
            # print(getattr(turbulence.mdata,name))
            if name == key_fix:
                # print("Load Mdata itself")
                getattr(self, 'load')(f[key_fix + key][keys[0]])
            else:
                obj = getattr(getattr(turbulence.mdata, name), name)(generate=False)
                setattr(self, name, obj)
                # print(name)
                getattr(getattr(self, name), 'load')(f[key_fix + key][keys[0]])
                # print("done")
        else:
            #   print("Iterate...")
            keys = f[key_fix + key].keys()
            if not keys == []:
                for k in keys:
                    self.load_rec(f, name, key=key + '/' + k)
                    # else:
                    #    print("done")
                    #  hdf5_Sdata = f['Mdata'][dpath.util.search(dict(f['Mdata']),'*mdata.Sdata*').keys()[0]]

    def add_param(self, name, unit):
        value = browse.get_number(self.Sdata.fileCine, '_' + name, unit + '_', from_end=True)
        setattr(self.param, name, value)
        # print("New attribute : "+name+'='+str(value))
        # self.write(data=True,overwrite=True)

    def get_filename(self, frmt=''):
        """
        Return the location where this class instance is (or will be) stored
        """
        # print(self.Id)
        return file_architecture.os_c(self.fileDir) + "M_" + self.Id.get_id() + frmt

    def get_fileList(self, display=False):
        """
        Return the list of PIV data (.txt files) associated to this measure
        """
        fileList, n = browse.get_fileList(self.dataDir + '/', self.frmt, display=display, sort='name')
        if n == 0:
            print("NO PIV files found locally at : " + self.dataDir)
            print("Look for other possible locations")

            rootDir = file_architecture.get_dir(self.Id.date)
            dataDir = browse.get_string(self.dataDir, self.Id.date, from_end=True)
            dataDir = rootDir + dataDir
            fileList, n = browse.get_fileList(dataDir + '/', self.frmt, display=display, sort='name')

            if n == 0:
                print("No PIV files found globally !")
            else:
                print("PIV files found at : " + rootDir)

        # print(n)
        return fileList, n

    def load_ref(self):
        pass

    def read_ref(self, Ref):
        # print(Ref)
        U_ref = {field: Ref[self.fieldnames[field]] for i, field in enumerate(self.fieldnames)}
        # }{'x':[],'y':[],'ux':[],'uy':[]}
        return U_ref

    def set_dimensions(self):
        fileList, nt = self.get_fileList()
        print(np.shape(self.U_ref['ux']))
        nx, ny = np.shape(self.U_ref['ux'])

        return nx, ny, nt

    def set_scales(self):
        # ft is in ms, time elapsed between two consecutive images
        ft = 1000. / self.param.fps
        # timescale is the time resolution of the PIV  measurements
        num = browse.get_number(self.dataDir, "_at", "fps")
        print(num)

        if num > 0:
            timescale = num * ft / 1000.
        else:
            timescale = 1

        fx = self.param.fx

        return ft, timescale, fx

    def shape(self):
        if not hasattr(self, 'nx'):
            self.U_ref = self.read_ref(self.load_ref())
            # U_ref has attribute x,y, Ux, Uy
            self.nx, self.ny, self.nt = self.set_dimensions()
            self.ft, self.timescale, self.fx = self.set_scales()
            #        self.x,self.y,self.t=set_scales()
            self.x, self.y = self.space_axis()
            self.t = self.time_axis()

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
            t0 = self.im_index[0] * self.ft
        else:
            print("No time reference found, origin of time arbitrary chosen at the beginning")
            t0 = 0.

        t = [i * self.ft - t0 for i in self.im_index]
        # time axis in ms
        # suppose that the frame are regularly spaced (!)
        # define it from the m_index list ??
        return t

    def space_axis(self):
        x = self.U_ref['x']
        y = self.U_ref['y']

        if not hasattr(self.param, 'x0') or self.param.x0 is None:
            # print("No space reference found, origin of space arbitrary chosen at the up left corner")
            x0 = 0.
            y0 = 0.
        else:
            x0 = self.param.x0
            y0 = self.param.y0

        x = (x - x0) * self.fx
        y = -(y - y0) * self.fx
        # if self.param.sIdeview=='bv'
        #    ...
        # if not (self.param.angle==0):
        #    pass
        #            print('rotation of the data needed !')
        return x, y

    ##################### Read and Write data ###########################

    def load_measure(self):
        if os.path.exists(self.get_filename(frmt='.hdf5')):
            print("loading from hdf5 file")
            self.load_all()
            return None

        fileList, nt = self.get_fileList(True)

        self.nt = nt
        self.t = self.time_axis()

        if nt == 0:
            return False
        # update = True means no change if the files are already present in the file
        # from a directory where all the .npz are stored, load_PIVData generate the x,y,Ux,Uy fields of Sdata
        # use the Dir name to find the fps ??
        # scale from the PIV settings : to be upgraded to a PIVsettings object
        # self.scale = 2.8 * 10**-5

        # save parameters here !
        print("Loading data ...")
        self.Ux = np.zeros(self.shape())
        self.Uy = np.zeros(self.shape())

        # additionnal processing on the velocity measurements for now due to wrong PIVsettings (fps ans xscale)
        fact_x = self.fx  # /self.scale
        fact_t = 1 / self.ft
        fact = fact_x * fact_t

        for name in fileList:
            count = fileList.index(name)
            S = np.load(name)
            # print(fileList.index(name))
            if nt > 100:
                if count % (nt // 100) == 0:
                    pass
            self.Ux[:, :, fileList.index(name)] = fact * S['ux']
            self.Uy[:, :, fileList.index(name)] = fact * S['uy']
            S.close()

        print("Data loaded")
        return True

    def eraseData(self):
        """Erase Ux and Uy data"""
        self.Ux = np.zeros((0, 0, 0))
        self.Uy = np.zeros((0, 0, 0))
        # Write the Sdata in the indicate filename

    def hdf5_exist(self):
        return os.path.exists(self.get_filename(frmt='.hdf5'))

    def write(self, data=False, overwrite=False):
        if self.hdf5_exist() and not overwrite:
            print("Already exist, no overwrite")
        else:
            if not hasattr(self, 'Ux'):
                print("load measure")
                self.load_measure()
            if not data:
                self.eraseData()

            print("Writing Measure in hdf5 file ")
            t1 = time.time()
            self.write_hdf5()
            t2 = time.time()
            print('Time elapsed : ' + str(round((t2 - t1) * 100) / 100.) + 's')
            print("Data written")

    def write_hdf5(self):
        filename = self.get_filename(frmt='.hdf5')
        to_hdf5.write(self, filename=filename, key='Mdata')

    def get_index(self, x0, y0):
        """
        Return the closest index from the requested position 
        should depend on the orientation : if 90 angle, i and j roles are inverted    
        """
        if self.param.angle == 90:
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

    ########################## Measurements #############################

    def get(self, field, **kwargs):
        """
        Field you can get from here :
        'U' : write Ux and Uy components into a single matric
        'E' : kinetic energy E = Ux^2 + Uy^2 (energy density, rho=1)
        'omega' : vorticity field
        'dU' : full strain tensor components. Return a numpy array of shape (nx,ny,nt,2,2)
        'Enstrophy' :
        """
        if field == 'U':
            # return both component in a vectorial format
            Ux = self.get('Ux')
            Uy = self.get('Uy')
            data = np.transpose(np.asarray([Ux, Uy]), (1, 2, 3, 0))
            return data
        # if (not hasattr(self,field)) or (compute):

        #            vgradient.compute(M,field,Dt=Dt_filt,**kwargs)
        if not hasattr(self, field):
            if 'Dt_filt' in kwargs and kwargs['Dt_filt'] > 1:
                print('Filtering of the data : irreversible')
            self.compute(field)
        # setattr(self,field,)
        if hasattr(self, field):
            return getattr(self, field)
        else:
            return None

    def get_cut(self):
        print("crazy")

    def compute(self, field, **kwargs):
        return vgradient.compute(self, field, **kwargs)

    def measure(self, name, function, force=False, *args, **kwargs):
        if (not hasattr(self, name)) or force:
            print("Compute " + name)
            val = function(self, *args, **kwargs)
            setattr(self, name, val)
        else:
            print("Already computed")

######################### Plots ############################
