# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:42:10 2015

@author: stephane
"""
import turbulence.tools.browse as browse
import turbulence.hdf5.h5py_convert as to_hdf5
import h5py


class Id:
    def __init__(self, S=None, generate=True, **kwargs):
        # list of parameters, should be a dictionnary ?
        # Identification number
        self.Sdata = S

        if generate:
            self.gen(**kwargs)

    def __name__(self):
        return 'Id'

    def load(self, data):
        self = to_hdf5.load(self, data)

    def gen(self, index=-1, mindex=0, typ="Accelerated grid", who='SPerrard'):
        self.index = index
        self.mindex = mindex
        self.type = typ
        self.who = who

        if hasattr(self.Sdata, 'fileCine'):
            self.set_date(self.Sdata.fileCine)
        else:
            self.date = ''

    def get_id(self):
        identifiant = self.date + "_" + str(self.index) + "_" + str(self.mindex)
        return identifiant

    def set_date(self, file):
        # Date
        #        file=self.Sdata.fileCine
        self.date = browse.get_string(file, "/20", "/", -2)
