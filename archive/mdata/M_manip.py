# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:57:41 2015

@author: stephane
"""

import os.path
import turbulence.tools.pickle_m as pickle_m
import turbulence.tools.browse as browse
import turbulence.manager.file_architecture as file_architecture
import turbulence.mdata.Sdata_manip as Sdata_manip
from turbulence.mdata.Mdata import Mdata
from turbulence.mdata.Mdata_PIVlab import Mdata_PIVlab
from turbulence.mdata.Mdata_pyPIV import Mdata_pyPIV


def Measure_gen_day(date):
    Slist = Sdata_manip.load_Sdata_day(date)
    print(Slist)

    #    print(Slist)
    for S in Slist:
        # try:
        Measure_gen_serie(S)
        # except:
        # print(dir(S))
        # print("Cine file : "+S.fileCine)
        # print('Fail to check processed PIV data')


def Measure_gen_serie(S):
    """look for every data files associated to this cine. the Dir name must contained explicitly cinefile name
    sorted by date of creation"""
    # generate a bunch of M objects
    Dir = os.path.dirname(S.fileCine)
    Dir = file_architecture.os_c(Dir)
    fileCine = Dir + '/' + os.path.basename(S.fileCine)

    print(fileCine)
    dataDirList, n = browse.get_dataDir(fileCine, root='/PIV_*data/PIV', display=True)
    #    dataDirList2,n=browse.get_dataDir(fileCine,root='/PIV_step1_data/PIV',display=True)
    #    dataDirList3,n=browse.get_dataDir(fileCine,root='/PIV_step10_data/PIV',display=True)
    # dataDirList3,n=browse.get_dataDir(fileCine,root='/PIV_data_full/PIV',display=True)
    #    dataDirList = dataDirList+dataDirList2+dataDirList3
    #   get_dataDir(cinefile,root='/PIV_',frmat='cine',disp=False)
    # print("")
    #  print(dataDirList)
    if n == 0:
        #  print(S.fileCine)
        print('This cinefile have not been analyzed')
    else:
        print(dataDirList)
        # input()

        for name in dataDirList:
            # browse.get_string()
            if True:  # '/PIV_W32_data/' in name: ### to generate only a subset of the Mdata (in particular in case of processing in progress)
                print(name, dataDirList.index(name))

                # How to find with which software the data have been analysed ?
                # M_gen(S,name,dataDirList.index(name),'pyPIV')
                # launch the appropriate M_gen options based on the name of the dataDirList

                print('')
                print('generate Mdata')
                print(dataDirList.index(name))
                print(name)
                print('')
                M_gen(S, name, dataDirList.index(name), 'PIVlab')
            else:
                print('')
                print('skip')
                print('')


def M_gen(S, dataDir, mindex, typ='Mdata'):
    if typ == 'Mdata':
        m = Mdata(S=S, generate=True, dataDir=dataDir, mindex=mindex)
    if typ == 'PIVlab':
        m = Mdata_PIVlab(S=S, generate=True, dataDir=dataDir, mindex=mindex)
    if typ == 'pyPIV':
        m = Mdata_pyPIV(S=S, generate=True, dataDir=dataDir, mindex=mindex)

    #    frmt = '.hdf5'
    #    formats = ['.hdf5','.txt']
    #    for frmt in formats:
    #        filename = m.get_filename(frmt=frmt)
    #        if os.path.exists()
    #            self.load
    #    m.load()
    m.write(data=True)
    # p2json.write_rec(M,erase=True)  #generate a json file containing the parameters
    return m


#    print(m)

def load(date, dataDir, data=False):
    """
    Load the data contained in a dataDir, in a Mdata object format
    INPUT
    -----
    date : str
        date to look at
        (could be found from dataDir, not implemented yet)
    dataDir : str
        path of the data directory
    OUTPUT
    -----
    M : Mdata object
        contains the data see class Mdata for details
    """
    Mlist = load_Mdata_serie(date, data=False)
    # look for the right
    for M in Mlist:
        if M.dataDir == dataDir:
            return M
    print('Data not found, check the name')
    return None


def load_Mdata_file(filename, data=True):
    # load a Sdata using its filename
    # print(filename)
    if os.path.isfile(filename):
        # print('Reading Mdata')
        M = Mdata_PIVlab(generate=False)  # create an empty instance of Mdata object
        setattr(M, 'filename', filename[:-5])  # set the reference to the filename where it was stored
        # print(M.filename)
        M.load_all()  # load all the parameters (only need an attribute filename in S)

        return M
    else:
        print('No data found for ' + S.filename)
        return None


def load_Mdata(date, index, mindex):
    filename = getloc(date, index, mindex)
    M = load_Mdata_file(filename[0])
    return M


# def load_Sdata_serie(date,indexList,mindexList):
#    Slist=[]
#    for index in indexList:
#        for mindex in mindexList[indexList.index(index)]:
#            Slist.append(load_Sdata(date,index,mindex))

#    return Slist
def load_Mdata_serie(date, index=[], mindex=[], data=True):
    # list with the different Sdata files
    Mlist = []
    fileList = getloc(date, index, mindex)
    for file in fileList:
        index = int(browse.get_string(file, 'M_' + date + '_', '_'))

        try:
            M = load_Mdata_file(file, data=data)
            done = True
        except:
            # retrieve the Sdata associated to it
            S = Sdata_manip.load_Sdata(date, index)
            if S is not None:
                Measure_gen_serie(S)
                M = load_Mdata_file(file, data)
                done = True
            else:
                done = False
            #            for i,M in enumerate(Mlist):
            #                M_gen(S,dataDir,mindex,typ='Mdata')
        if done:
            Mlist.append(M)
        else:
            print("")
            print("unable to read Sdata for " + date)
            print("")

    return Mlist
    # Sdata_measure.multi_mean_profile(SList)


def read_hdf5(filename):
    pass


#    pass

def read(filename, data=False):
    M = Mdata(generate=False)  # create an empty instance of Sdata object
    setattr(S, 'filename', filename[:-5])  # set the reference to the filename where it was stored
    #   print(S.filename)
    S.load_all()  # load all the parameters (only need an attribute filename in S)

    m = pickle_m.read(filename)
    if data:
        m.load_measure()
    return m


def getloc(date, index, mindex):
    Dir = getDir(date)
    # print(Dir)
    if index == []:
        print('load every Mdata')
        fileList, n = browse.get_fileList(Dir, 'txt', root='M_')
        print(fileList)
    else:
        if mindex == []:
            print('load every Mdata associated to ' + date + '_' + str(index))
            fileList, n = browse.get_fileList(Dir, 'txt', root='M_' + date + str(index))
        else:
            filename = Dir + "M_" + date + "_" + str(index) + "_" + str(mindex) + ".txt"
            fileList = [filename]
    return fileList


def getloc_M(m):
    Dir = getDir(m.id.date)
    filename = Dir + "M_" + m.id.date + "_" + str(m.id.index) + "_" + str(m.id.mindex) + ".txt"
    return filename


def getDir(date):
    rootdir = file_architecture.get_dir(date)
    Dir = rootdir + "/M_" + date + "/"
    return Dir
