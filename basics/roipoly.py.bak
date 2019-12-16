import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.path as mplPath

'''Draw polygon regions of interest (ROIs) in matplotlib images,
similar to Matlab's roipoly function.

See the file example.py for an application. 

Created by Joerg Doepfert 2014 based on code posted by Daniel
Kornhauser. Some improvements by Noah Mitchell 2017.

'''


class RoiPoly:
    def __init__(self, fig=None, ax=None, roicolor='b'):
        if not fig:
            fig = plt.gcf()

        if not ax:
            ax = plt.gca()

        self.previous_point = []
        self.allxpoints = []
        self.allypoints = []
        self.start_point = []
        self.end_point = []
        self.line = None
        self.roicolor = roicolor
        self.fig = fig
        self.ax = ax
        # self.fig.canvas.draw()

        self.__ID1 = self.fig.canvas.mpl_connect('motion_notify_event', self.__motion_notify_callback)
        self.__ID2 = self.fig.canvas.mpl_connect('button_press_event', self.__button_press_callback)

        if sys.flags.interactive:
            plt.show(block=False)
        else:
            plt.show()

    def get_mask(self, currentImage):
        ny, nx = np.shape(currentImage)
        poly_verts = [(self.allxpoints[0], self.allypoints[0])]
        for i in range(len(self.allxpoints) - 1, -1, -1):
            poly_verts.append((self.allxpoints[i], self.allypoints[i]))

        # Create vertex coordinates for each grid cell...
        # (<0,0> is at the top left of the grid in this system)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        ROIpath = mplPath.Path(poly_verts)
        grid = ROIpath.contains_points(points).reshape((ny, nx))
        return grid

    def displayROI(self, **linekwargs):
        l = plt.Line2D(self.allxpoints +
                       [self.allxpoints[0]],
                       self.allypoints +
                       [self.allypoints[0]],
                       color=self.roicolor, **linekwargs)
        ax = plt.gca()
        ax.add_line(l)
        plt.draw()

    def display_mean(self, currentImage, **textkwargs):
        mask = self.get_mask(currentImage)
        meanval = np.mean(np.extract(mask, currentImage))
        stdval = np.std(np.extract(mask, currentImage))
        string = "%.3f +- %.3f" % (meanval, stdval)
        plt.text(self.allxpoints[0], self.allypoints[0],
                 string, color=self.roicolor,
                 bbox=dict(facecolor='w', alpha=0.6), **textkwargs)

    def __motion_notify_callback(self, event):
        if event.inaxes:
            ax = event.inaxes
            x, y = event.xdata, event.ydata
            # Move line around
            if (event.button is None or event.button == 1) and self.line is not None:
                self.line.set_data([self.previous_point[0], x],
                                   [self.previous_point[1], y])
                self.fig.canvas.draw()

    def __button_press_callback(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            ax = event.inaxes
            # If you press the left button, single click
            if event.button == 1 and not event.dblclick:
                # if there is no line, create a line
                if self.line is None:
                    self.line = plt.Line2D([x, x],
                                           [y, y],
                                           marker='o',
                                           color=self.roicolor)
                    self.start_point = [x, y]
                    self.previous_point = self.start_point
                    self.allxpoints = [x]
                    self.allypoints = [y]

                    ax.add_line(self.line)
                    self.fig.canvas.draw()
                    # add a segment
                else:
                    # if there is a line, create a segment
                    self.line = plt.Line2D([self.previous_point[0], x],
                                           [self.previous_point[1], y],
                                           marker='o', color=self.roicolor)
                    self.previous_point = [x, y]
                    self.allxpoints.append(x)
                    self.allypoints.append(y)

                    event.inaxes.add_line(self.line)
                    self.fig.canvas.draw()
            elif ((event.button == 1 and event.dblclick) or
                      (event.button == 3 and not event.dblclick)) and self.line is not None:
                # close the loop and disconnect
                self.fig.canvas.mpl_disconnect(self.__ID1)
                self.fig.canvas.mpl_disconnect(self.__ID2)

                self.line.set_data([self.previous_point[0],
                                    self.start_point[0]],
                                   [self.previous_point[1],
                                    self.start_point[1]])
                ax.add_line(self.line)
                self.fig.canvas.draw()
                self.line = None

                if sys.flags.interactive:
                    pass
                else:
                    # figure has to be closed so that code can continue
                    plt.close(self.fig)


if __name__ == '__main__':
    import pylab as pl

    # create image
    img = pl.ones((100, 100)) * range(0, 100)

    # show the image
    pl.imshow(img, interpolation='nearest', cmap="Greys")
    pl.colorbar()
    pl.title("left click: line segment         right click: close region")

    # let user draw first ROI
    ROI1 = RoiPoly(roicolor='r')

    # show the image with the first ROI
    pl.imshow(img, interpolation='nearest', cmap="Greys")
    pl.colorbar()
    ROI1.displayROI()
