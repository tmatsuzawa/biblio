#!/usr/bin/env python
import argparse
import json
import os
import pickle
import sys
import time
import glob
import sparse
from fmm import *
from ilpm import path, vector
from matplotlib import cm
from numpy import *
import matplotlib.pyplot as plt



class Clicker:
    """A Class used to store information about click positions and display them on a figure."""

    def __init__(self, ax):
        self.canvas = ax.get_figure().canvas
        self.cid = None
        self.pt_lst = []
        self.pt_plot = ax.scatter([], [], marker='o', color='m')
        self.connect_sf()

    def clear(self):
        """Clears the points"""
        self.pt_lst = []
        self.redraw()

    def connect_sf(self):
        if self.cid is None:
            self.cid = self.canvas.mpl_connect('button_press_event', self.click_event)

    # def disconnect_sf(self):
    #     if self.cid is not None:
    #         self.canvas.mpl_disconnect(self.cid)
    #         self.cid = None

    def click_event(self, event):
        """ Extracts locations from the user"""
        if event.key == 'shift':
            # Shift + click to remove all selections
            print('cleared click events')
            self.clear()
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            print('click event: xdata = %f, ydata= %f' % (event.xdata, event.ydata))
            self.pt_lst.append((event.xdata, event.ydata))
        elif event.button == 3:
            # Right click to remove selected pt
            print('removed click event near: data = %f, ydata = %f' % (event.xdata, event.ydata))
            self.remove_pt((event.xdata, event.ydata))

        self.redraw()

    def remove_pt(self, loc):
        """ Removes point from pt_lst that is nearest to loc"""
        if len(self.pt_lst) > 0:
            self.pt_lst.pop(argmin([sqrt((x[0] - loc[0]) ** 2 + (x[1] - loc[1]) ** 2) for x in self.pt_lst]))

    def redraw(self):
        """ Scatter points from pt_lst onto the figure"""
        if len(self.pt_lst) > 0:
            pts = asarray(self.pt_lst)
        else:
            pts = []

        self.pt_plot.set_offsets(pts)
        self.canvas.draw()