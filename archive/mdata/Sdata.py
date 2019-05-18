import os.path
# import h5py
import dpath
import turbulence.mdata
import turbulence.mdata.Id as Id
import turbulence.mdata.param as param
import turbulence.tools.pickle_m as pickle_m
import turbulence.hdf5.h5py_convert as to_hdf5
import glob
import turbulence.manager.file_architecture as file_architecture
import sys

'''Define the Sdata class, which contains
parameter, id, and association to its physical location
The Sdata class has the following attributes:
Id
param
fileDir
fileCine
dirCine

__init__ file calls a gen() method to instantiate the attributes. This is bad but we can't remove it without changing
lots of calls to this class. -npm 20170723
'''


class Sdata(object):
    """
    """
    def __init__(self, generate=True, **kwargs):
        # Physical location
        # of the raw data, of the measure, and of the present object (in pickle file)
        # filename of the associated cine file (without the dir!)
        if generate:
            # initializing attributes outside __init__ is bad form... for now we do it here -npm
            self.gen(**kwargs)
        else:  # generate an empty object
            self.Id = Id.Id(self, index=-1, mindex=0)
            self.param = param.param(self, generate=False)
            print("initialized")

    def __name__(self):
        return 'Sdata'

    def gen(self, fileCine='', index=-1, mindex=0, fileParam=''):
        """This method actually initializes the class instance. This is a horrible way to do this, since attributes
        are created outside the __init__ method. We need to get rid of any time when Sdata(generate=False) is used
        elsewhere and move the content of this method to the __init__() method.
        Sdata contains only the parameter, id, and association to its physical location.

        Parameters
        ----------
        fileCine :
        index :
        mindex :
        fileParam :

        Returns
        -------
        """
        self.fileCine = fileCine
        self.dirCine = self.read_dir()

        # Identification
        self.Id = Id.Id(self, index=index, mindex=mindex)
        self.param = param.param(self, generate=False)
        self.fileDir = self.dirCine + "Sdata_" + self.Id.date + "/"

        if os.path.exists(self.get_filename('.hdf5')):
            self.load_all(source=self.get_filename('.hdf5'))
            return None
        elif os.path.exists(self.get_filename('.txt')):
            try:
                self.load_all(source=self.get_filename('.txt'))
                return None
            except:
                print("Pickle object cannot be read anymore")
                pass
        # Parameters
        self.param = param.param(self)
        self.param.load_exp_parameters(fileParam)
        self.param.set_param()

        self.save_param()
        # Measurements
        # no more measure associated to the Sdata : it contains only the parameter, id, and association to
        # its physical location

        # self.m=measure.M(self,dirData)

    def get_filename(self, frmt):
        return self.fileDir + "Sdata_" + self.Id.get_id() + frmt

    # object  #type of measurement used #future attibutes of a PIV measurement type
    # Identification of the associated PIV_measurements ?
    # -> several set of measurement should be done on the same initial cine file !
    # measurement has to be indexed

    def load(self, data):
        #  print(data.keys())
        self = to_hdf5.load(self, data)

        #  print(self.fileCine)

    def load_all(self, source=None):
        """Load the data from various possible sources :
        pickle file
        json file
        h5py
        from scracth (generate everything from Sdata)
        """
        if source is None and hasattr(self, 'filename'):
            filename = self.filename + '.hdf5'
        else:
            filename = source
            if filename is None:
                print("No hdf5 file found")

        print 'Sdata.load_all(): Attempting to load from hdf5...'
        fi = to_hdf5.open(filename)
        if 'Id' in fi['Sdata'].keys():
            print "Sdata.load_all(): loading with self.load(fi['Sdata']) since key 'Id' is in hdf5['Sdata']..."
            self.load(fi['Sdata'])
            print("Sdata.load_all(): self.fileCine = " + self.fileCine)
        else:
            print "Sdata.load_all(): loading with self.load_rec() using a list [Sdata, param, Id]..."
            names = ['Sdata', 'param', 'Id']
            for name in names:
                self.load_rec(fi, name)
                #    print(self.param)
        fi.close()

    def load_measures(self, frmt='.hdf5'):
        """Look for measurements associated to this cine file, and if so, load them in Mdata objects,
            and return a list of the Mdata objects.
            Should it be inserted direcly in the Sdata object ?
            -> It could be problematic for the storage, but simpler from a coding point of view.
        """
        import turbulence.mdata.M_manip as M_manip
        searchstr = file_architecture.os_c(self.dirCine) + 'M_' + self.Id.date + '/M_' + self.Id.date + '_' +\
                    str(self.Id.index) + '_*' + frmt
        print('Sdata.Sdata.load_measures(): Looking for ' + searchstr + '...')
        filelist_m = glob.glob(searchstr)
        if not filelist_m:
            print 'filelist_m is empty'
            sys.exit()

        mlist = []
        # print("Data processed found at : "+str(fileList_M))
        for filename in filelist_m:
            print('Sdata.Sdata.load_measures(): filename = ' + filename)
            mlist.append(M_manip.load_Mdata_file(filename, data=True))
            print 'Sdata.Sdata.load_measures(): mlist = ', mlist
        return mlist

    def load_measure(self, indice=0, frmt='.hdf5'):
        import turbulence.mdata.M_manip as M_manip

        filename = file_architecture.os_c(self.dirCine) + 'M_' + self.Id.date + '/M_' + self.Id.date + '_' + str(
            self.Id.index) + '_' + str(indice) + frmt

        if os.path.exists(filename):
            M = M_manip.load_Mdata_file(filename, data=True)
        else:
            print('file does not exist')
        return M

    def load_rec(self, f, name, key=''):
        key_fix = 'Sdata'
        # print(key_fix+key)
        # try:
        #    print("toto : " +str(dpath.util.search(dict(f[key_fix+key]),'*mdata.'+name+'*').keys()))
        keys = dpath.util.search(dict(f[key_fix + key]), '*mdata.' + name + '*').keys()
        # except:
        #    print("Nothing to look for anymore")
        #    return None
        if not keys == []:
            if name == key_fix:
                getattr(self, 'load')(f[key_fix + key][keys[0]])
            else:
                obj = getattr(getattr(turbulence.mdata, name), name)(generate=False)
                setattr(self, name, obj)
                getattr(getattr(self, name), 'load')(f[key_fix + key][keys[0]])
                #   print("done")
        else:
            #  print("Iterate...")
            keys = f[key_fix + key].keys()
            if not keys == []:
                for k in keys:
                    self.load_rec(f, name, key=key + '/' + k)
                    #   else:
                    #    print("done")

    def get_id(self):
        return self.Id.get_id()

    def read_dir(self):
        return self.fileCine[:str.rfind(self.fileCine, "/") + 1]

    def save_param(self):
        print(self.param.fileParam)
        # look for the param file
        print(self.param.Sdata.fileCine)

        # save the experimental parameters in a Ref file (smart !)
        # thwen the param file in pickle format will become obsolete
        #   if not os.path.exists(self.param.fileParam):
        #   print("Do not pickle anymore !")
        #       pickle_m.write(self,self.param.fileParam)
        #   else:
        #       print("already exist, skip")

    def read_param(self):
        """

        Returns
        -------

        """
        print('Reading Parameters from param file')
        # try to read a Sdata and put it in a param object !!
        fileParam = self.param.fileParam

        print(fileParam)
        try:
            S = pickle_m.read(fileParam)
        except:
            S = None
            print("Pickle file cannot be loaded anymore. Skip")
        print(self.param)
        # load an old version of the parameters : must test the type !
        if hasattr(S, 'param'):
            self.param = S.param
            # self.param.angle=90
        else:
            # update the parameters file from the old Sdata format
            self.param.fps = S.fps
            self.param.fx = S.fx
            self.param.H0 = S.H0
            self.param.stroke = S.stroke
            self.param.im0 = S.im0
            self.param.x0 = S.x0
            self.param.y0 = S.y0

            if hasattr(S, 'angle'):
                self.param.angle = S.angle
            else:
                print('No attribue angle,arbitrary set to 90')
                self.param.angle = 90
        self.param.Sdata = self
        self.param.fileParam = fileParam
        print(self.param.fileParam)

    def write(self):
        """Write the Sdata in the indicated filename -- depricated!

        """
        # Depreciated
        print("Writing Sdata in hdf5 file")

        self.im_ref = None
        # write the Sdata object in a .txt file using the package pickle
        # The filename is directly determined from the Identification number
        # default name for the file containing the data
        self.filename = self.fileDir + "Sdata_" + self.Id.get_id()
        self.write_hdf5()
        #        pickle_m.write(self,self.filename+'.txt')

        print("Data written")

    def write_hdf5(self):
        filename = self.filename + '.hdf5'

        if not os.path.exists(filename):
            to_hdf5.write(self, filename=filename, key='Sdata')
        else:
            print("need to be updated ? Most likely no ! Skip")
            # Read the Sdata object from the indicate filename
            # static method, generate a Sdata from the loaded file
            #   @staticmethod
            # def load_measure(self):
            #    self.m.loadData()

            # def update(self,Dir_PIV):
            # compare the current dimensions of the data (must be stored in the parameter file for a better efficiency!)
            # With the number of files in Dir
            # check for update
            #    self.m.update()

            #    nx,ny,nt=self.shape()

            #    fileList,n=browse.get_fileList(Dir_PIV+"/","npz")

            #    if n==nt:
            #        print("data already present, no correction needed")
            #        return False
            #    else:
            #        self.load_PIVData(fileList,Dir_PIV)
            #        return True

            # Load the data computed from PIV measurements, stored in a list of npz files
            # now demands the location of the associated file
