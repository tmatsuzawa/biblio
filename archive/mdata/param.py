# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:42:10 2015

@author: stephane
"""

import os.path
import matplotlib.pyplot as plt
import numpy as np

import turbulence.cine as cine
import turbulence.mdata.initial_time as intial_time
import turbulence.tools.rw_data as rw_data
import turbulence.tools.browse as browse

import turbulence.hdf5.h5py_convert as to_hdf5
import h5py

import glob


class param:
    def __init__(self, S=None, generate=True):
        # list of parameters, should be a dictionnary ?
        # will depend on the considered object. for now, the type is moving grid
        # could be read from the title ?
        self.Sdata = S

        if generate:
            self.fileParam = self.filename_param('r')

    def __name__(self):
        return 'param'

        # Experimental condition

    def load(self, data):
        self = to_hdf5.load(self, data)

    def set_param(self):
        file = self.Sdata.fileCine

        self.Xplane = browse.get_number(file, "_X", "mm", from_end=False)
        self.Zplane = browse.get_number(file, "_Z", "mm", from_end=False)
        self.fps = browse.get_number(file, "_fps", "_", from_end=False)
        self.H0 = browse.get_number(file, "_H", "mm", from_end=False)
        self.stroke = browse.get_number(file, "_S", "mm", from_end=False)

        if file.find('_sv') > 0:
            self.typeview = 'sv'
        else:
            if file.find('_bv') > 0:
                self.typeview = 'bv'
            else:
                self.typeview = 'NA'

        if file.find('_hp') > 0:
            self.typeplane = 'hp'
        else:
            if file.find('_vp') > 0:
                self.typeplane = 'vp'
            else:
                self.typeplane = 'NA'

            #        print(self.typeplane)

        if (not hasattr(self, 'im0')) or (self.im0 == 0):
            pass
            #     self.get_im0()

    def load_exp_parameters(self, fileParam=''):
        self.fileParam = self.fileParam + '.txt'
        self.im_ref = self.read_reference(ref=0)

        # default value of fileParam :
        if True:  # fileParam == '':
            base = browse.get_string(os.path.basename(self.Sdata.fileCine), '', end='.cine')
            #   print(base)
            fileRef = os.path.dirname(self.Sdata.fileCine) + '/Setup_file_Reference*.txt'
            #   print(fileRef)
            fileList = glob.glob(fileRef)

            print(base)
            #  print(fileList)
            for filename in fileList:
                pattern = browse.get_string(filename, 'Setup_file_Reference', end='.txt')
                print(pattern)

                if pattern in base:
                    fileParam = filename
                    # find the file where exp. parameters are given
                    #   fileParam = os.path.dirname(self.Sdata.fileCine)+'/References/Ref_'+base+'.txt'

        # other name :
        print(fileParam)

        if os.path.isfile(fileParam):
            # read the parameters from the param file
            Header, Data = rw_data.read_dataFile(fileParam, Hdelimiter='\t', Ddelimiter='\t')
            print(Data)

            for i in Data.keys():
                setattr(self, i, Data[i][0])
        else:
            if os.path.isfile(self.fileParam):
                self.im_ref = self.read_reference(ref=0)
                self.Sdata.read_param()
                # load parameters from the parameter file
            else:
                self.im_ref = self.read_reference(ref=0)

                self.manual_input()

    # self.generate_parameter_file()

    def filename_param(self, opt="w"):
        if opt == "r":
            name = self.Sdata.fileCine[:-5] + "_param"  # without the cine extension
        else:
            #            print(self.Sdata.fileCine)
            name = "PIV_" + self.Sdata.get_id() + "_param"
            name = self.Sdata.dirCine + name
        return name

    def manual_input(self):
        # load the first image and save it in a reference file
        print(
        "Manual loading of experimental parameters for " + self.Sdata.fileCine)  # self.date + " ," + str(self.fps) +" fps")
        self.fx = float(input("fx = "))
        self.im0 = float(input("im(t=0) = "))
        self.x0 = float(input("x0 = "))
        self.y0 = float(input("y0 = "))
        self.angle = float(input("angle = "))
        # generate a param file (in case of the data have to be reloaded)
        # name of the file :
        # Rotating angle

    def get_im0(self):
        self.im0 = initial_time.get_im0(self, int(self.im0))

    def comments(self):
        self.description = input("")

    def read_reference(self, ref=0):
        print(self.Sdata.fileCine)
        try:
            c = cine.Cine(self.Sdata.fileCine)
        except:
            print('Cine file unavailable')
            return None
        print('cine open')
        # get maximum value according to bit depth
        bitmax = float(2 ** c.real_bpp)
        # get frames from cine file
        a = c[ref].astype("f")
        a = a / bitmax * 255

        im_ref = np.float64(a)

        print(np.shape(im_ref))

        plt.figure(1)
        plt.imshow(im_ref)
        #        graphes.refresh()
        plt.show(False)
        # save the figure
        plt.draw()

        c.close()

        print('cine closed')

        return im_ref
