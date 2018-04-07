import numpy as np
import os
from math import *
import sys
sys.path.append('/Users/stephane/Documents/git/takumi/library/')
# import library.tools.fitting as fitting
import library.hdf5.h5py_s as h5py_s

'''
Module for plotting and saving data
Many methods were taken from Stephane's codes,  and I am going to customize it. - Takumi 10/24/17
'''

# Define global variables
__fontsize__ = 12

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools

#Global variables
#Default color cycle: iterator which gets repeated if all elements were exhausted
__color_cycle__ = itertools.cycle(iter(plt.rcParams['axes.prop_cycle'].by_key()['color']))
__fontsize__ = 16
__figsize__ = (8, 8)

## Save a figure
def save(path, ext='png', close=False, verbose=True, fignum=None, **kwargs):
    """Save a figure from pyplot
    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.
    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.
    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.
    """
    if fignum == None:
       fignum = plt.gcf()
    else:
        fignum = plt.figure(fignum)


    # Separate a directory and a filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # path where the figure is saved
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving figure to '%s'..." % savepath),

    # Save the figure
    plt.savefig(savepath)

    # Close it
    if close:
        plt.close()

    if verbose:
        print("... Done")


## Create a figure and axes
def set_fig(fignum, subplot=None, dpi=100, figsize=__figsize__, **kwargs):
    if fignum == -1:
        fig = plt.figure(dpi=dpi, figsize=figsize)
    if fignum == 0:
        fig = plt.cla()  #clear axis
    if fignum > 0:
        fig = plt.figure(num=fignum, dpi=dpi, figsize=figsize)
        fig.set_size_inches(figsize[0], figsize[1])
        fig.set_dpi(dpi)

    if subplot is not None:
        # a triplet is expected !
        ax = fig.add_subplot(subplot, **kwargs)
        return fig, ax
    else:
        ax = fig.add_subplot(111)
        return fig, ax


def plotfunc(fun, x, param, fignum=1, label='-', color = None, subplot=None, legend=False, figsize=__figsize__, **kwargs):
    """
    plot a graph using the function fun
    fignum can be specified
    any kwargs from plot can be passed
    Use the homemade function refresh() to draw and plot the figure, no matter the way python is called (terminal, script, notebook)
    """

    fig, ax = set_fig(fignum, subplot, figsize=figsize)
    y = fun(x, param)
    if not color==None:
        plt.plot(x, y, color=color, label=label, **kwargs)
    else:
        plt.plot(x, y, label=label, **kwargs)
    if legend:
        plt.legend()
    return fig, ax

def plot(x, y, fignum=1, figsize=__figsize__, label='-', color=None, subplot=None, legend=False, **kwargs):
    """
    plot a graph using given x,y
    fignum can be specified
    any kwargs from plot can be passed
    """
    fig, ax = set_fig(fignum, subplot, figsize=figsize)

    if len(x) > len(y):
        print("Warning : x and y data do not have the same length")
        x = x[:len(y)]
    if color is None:
        plt.plot(x, y, label=label, **kwargs)
    else:
        plt.plot(x, y, color=color, label=label,**kwargs)
    if legend:
        plt.legend()
    return fig, ax


def scatter(x, y, fignum=1, figsize=__figsize__, marker='o', label='-', color = next(__color_cycle__), subplot=None, legend=False, **kwargs):
    """
    plot a graph using given x,y
    fignum can be specified
    any kwargs from plot can be passed
    Use the homemade function refresh() to draw and plot the figure, no matter the way python is called (terminal, script, notebook)
    """
    fig, ax = set_fig(fignum, subplot, figsize=figsize)
    x, y = np.array(x), np.array(y)
    if len(x.flatten()) > len(y.flatten()):
        print("Warning : x and y data do not have the same length")
        x = x[:len(y)]
    ax.scatter(x, y, color=color, label=label, marker=marker, **kwargs)
    if legend:
        plt.legend()
    return fig, ax



def errorbar(x, y, xerr=0, yerr=0, fignum=1, fmt='o',label='-', color ='k', subplot=None, legend=False, figsize=__figsize__,**kwargs):
    """ errorbar plot

    Parameters
    ----------
    x : array-like
    y : array-like
    xerr: must be a scalar or numpy array with shape (N,1) or (2, N)... [xerr_left, xerr_right]
    yerr:  must be a scalar or numpy array with shape (N,) or (2, N)... [yerr_left, yerr_right]
    fignum
    label
    color
    subplot
    legend
    kwargs

    Returns
    -------
    fig
    ax

    """
    fig, ax = set_fig(fignum, subplot, figsize=figsize, **kwargs)
    # Make sure that xerr and yerr are numpy arrays
    ## x, y, xerr, yerr do not have to be numpy arrays. It is just a convention. - takumi 04/01/2018
    x, y = np.array(x), np.array(y)
    # Make xerr and yerr numpy arrays if they are not scalar. Without this, TypeError would be raised.
    if not (isinstance(xerr, int) or isinstance(xerr, float)):
        xerr = np.array(xerr)
    if not (isinstance(yerr, int) or isinstance(yerr, float)):
        yerr = np.array(yerr)

    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=fmt, label=label)
    if legend:
        ax.legend()
    return fig, ax

def errorfill(x, y, yerr, fignum=1, color=None, subplot=None, alpha_fill=0.3, ax=None, label='-',
              legend=False, figsize=__figsize__, color_cycle=__color_cycle__, **kwargs):
    fig, ax = set_fig(fignum, subplot, figsize=figsize)

    x = np.array(x)
    y = np.array(y)
    yerr = np.array(yerr)

    #ax = ax if ax is not None else plt.gca()
    if color is None:
        color = color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        yerrdown, yerrup = yerr
        ymin = y - yerrdown
        ymax = y + yerrup


    ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

    #patch used for legend
    color_patch = mpatches.Patch(color=color, label=label)
    if legend:
        plt.legend(handles=[color_patch])


    return fig, ax, color_patch


## 2D plots
def color_plot(x, y, z, subplot=None, fignum=1, vmin=0, vmax=0, figsize=__figsize__, log10=False, show=False, cbar=False, cmap='jet'):
    """  Color plot of 2D array
    Parameters
    ----------
    x 2d array eg. x = np.mgrid[slice(1, 5, dx), slice(1, 5, dy)]
    y 2dd array
    z 2d array
    subplot
    fignum
    vmin
    vmax
    log10
    show
    cbar
    cmap

    Returns
    -------
    fig
    ax
    cc QuadMesh class object

    """
    fig, ax = set_fig(fignum, subplot, figsize)

    if log10:
        z = np.log10(z)

    # Note that the cc returned is a matplotlib.collections.QuadMesh
    # print('np.shape(z) = ' + str(np.shape(z)))
    if vmin == vmax == 0:
        # plt.pcolormesh returns a QuadMesh class object.
        cc = plt.pcolormesh(x, y, z, cmap=cmap)
    else:
        cc = plt.pcolormesh(x, y, z, cmap=cmap, vmin=vmin, vmax=vmax)

    if cbar:
        plt.colorbar()

    return fig, ax, cc

def imshow(griddata, xmin=0, xmax=1, ymin=0, ymax=1, cbar=True, vmin=0, vmax=0, \
           fignum=1, subplot=111, figsize=__figsize__, interpolation='linear', cmap='bwr'):
    fig, ax = set_fig(fignum, subplot, figsize=figsize)
    if vmin == vmax == 0:
        cax = ax.imshow(griddata, extent=(xmin, xmax, ymin, ymax),\
                   interpolation=interpolation, cmap=cmap)
    else:
        cax = ax.imshow(griddata, extent=(xmin, xmax, ymin, ymax),\
                   interpolation=interpolation, cmap=cmap, vmin=vmin, vmax=vmax)
    if cbar:
        cc = fig.colorbar(cax)
    return fig, ax, cax, cc

def show():
    plt.show()

## Lines
def axhline(ax, y, x=None, **kwargs):
    """
    Draw a horizontal line at y=y from xmin to xmax
    Parameters
    ----------
    y
    x

    Returns
    -------

    """
    if x is not None:
        xmin, xmax = np.max(x), np.min(x)
    else:
        xmin, xmax= ax.get_xlim()
    ax.axhline(y, xmin, xmax, **kwargs)

def axvline(ax, x, y=None, **kwargs):
    """
    Draw a vertical line at x=x from ymin to ymax
    Parameters
    ----------
    x
    y

    Returns
    -------

    """
    if y is not None:
        ymin, ymax = np.max(y), np.min(y)
    else:
        ymin, ymax= ax.get_ylim()
    ax.axvline(x, ymin, ymax, **kwargs)


## Legend
# Legend
def legend(ax, **kwargs):
    """
    loc:
    best	0, upper right	1, upper left	2, lower left	3, lower right	4, right	5,
    center left	6, center right	7, lower center	8, upper center	9, center	10
    Parameters
    ----------
    ax
    kwargs

    Returns
    -------

    """
    ax.legend(**kwargs)


# Colorbar
def colorbar(fignum=plt.gcf().number, label=None, fontsize=__fontsize__):
    """

    Parameters
    ----------
    fignum :
    label :

    Returns
    -------
    """
    fig, ax = set_fig(fignum)
    c = plt.colorbar()
    if not label==None:
        c.set_label(label, fontsize=fontsize)
    return c


### Axes
# Label
def labelaxes(ax, xlabel, ylabel, fontsize=__fontsize__, **kwargs):
    ax.set_xlabel(xlabel, fontsize=fontsize, **kwargs)
    ax.set_ylabel(ylabel, fontsize=fontsize, **kwargs)


# Limits
def setaxes(ax, xmin, xmax, ymin, ymax):
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    return ax

## Set axes to semilog or loglog
def tosemilogx(ax=None):
    if ax == None:
        ax = plt.gca()
    ax.set_xscale("log", nonposx='clip')
def tosemilogy(ax=None):
    if ax == None:
        ax = plt.gca()
    ax.set_yscale("log", nonposx='clip')
def tologlog(ax=None):
    if ax == None:
        ax = plt.gca()
    ax.set_xscale("log", nonposx='clip')


##Title
def title(title, subplot=111):
    plt.subplot(subplot)
    plt.title(title, fontsize=__fontsize__)

def suptitle(title):
    plt.suptitle(title)




##Text
def set_standard_pos(ax):
    """
    Sets standard positions for added texts in the plot
    left: 0.1, right: 0.75
    bottom: 0.25 top: 0.75
    xcenter: 0.5 ycenter:0.5
    Parameters
    ----------
    ax

    Returns
    -------
    top, bottom, right, left, xcenter, ycenter: float, position

    """
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    width, height = np.abs(xright - xleft), np.abs(ytop - ybottom)

    left, right = xleft + 0.1 * width, xleft + 0.75 * width
    bottom, top = ybottom + 0.25 * height, ybottom + 0.75 * height
    xcenter, ycenter = width/2., height/2.
    return top, bottom, right, left, xcenter, ycenter


def addtext(ax, text='text goes here', x=0, y=0, fontsize=__fontsize__, color='k',
            option=None, **kwargs):
    """
    Adds text to a plot. You can specify the position where the texts will appear by 'option'
    Parameters
    ----------
    ax
    subplot
    text
    x
    y
    fontsize
    color
    option
    kwargs

    Returns
    ax : with a text
    -------

    """
    top, bottom, right, left, xcenter, ycenter = set_standard_pos(ax)

    if option == None:
        ax.text(x, y, text, fontsize=fontsize, color=color)
    if option == 'tr':
        ax.text(right, top, text, fontsize=fontsize, color=color)
    if option == 'tl':
        ax.text(left, top, text, fontsize=fontsize, color=color)
    if option == 'tc':
        ax.text(xcenter, top, text, fontsize=fontsize, color=color)
    if option == 'br':
        ax.text(right, bottom, text, fontsize=fontsize, color=color)
    if option == 'br2':
        ax.text(right, bottom/0.1*0.175, text, fontsize=fontsize, color=color)
    if option == 'br3':
        ax.text(right, bottom/0.1*0.10, text, fontsize=fontsize, color=color)
    if option == 'bl':
        ax.text(left, bottom, text, fontsize=fontsize, color=color)
    if option == 'bl2':
        ax.text(left, bottom/0.1*0.175, text, fontsize=fontsize, color=color)
    if option == 'bl3':
        ax.text(left, bottom/0.1*0.10, text, fontsize=fontsize, color=color)
    if option == 'bc':
        ax.text(xcenter, bottom, text, fontsize=fontsize, color=color)
    if option == 'cr':
        ax.text(right, ycenter, text, fontsize=fontsize, color=color)
    if option == 'cl':
        ax.text(left, ycenter, text, fontsize=fontsize, color=color)
    if option == 'cc':
        ax.text(xcenter, ycenter, text, fontsize=fontsize, color=color)
    return ax




##Clear plot
def clf(fignum=plt.gcf().number):
    plt.figure(fignum)
    plt.clf()

## Color cycle
def skipcolor(numskip, color_cycle=__color_cycle__):
    """ Skips numskip times in the color_cycle iterator
        Can be used to reset the color_cycle"""
    for i in range(numskip):
        color_cycle.next()
def countcolorcycle(color_cycle = __color_cycle__):
    return sum(1 for color in color_cycle)





######################
######################
######################
######################
######################
######################

# def plot(fun, x, y, fignum=1, label='-', subplot=None, **kwargs):
#     """
#     plot a graph using the function fun
#     fignum can be specified
#     any kwargs from plot can be passed
#     Use the homemade function refresh() to draw and plot the figure, no matter the way python is called (terminal, script, notebook)
#     """
#
#
#     set_fig(fignum, subplot=subplot)
#     y = fun(x, y, **kwargs)
#     refresh()

def graph(x, y, fignum=1, label='-', subplot=None, **kwargs):
    """
    plot a graph using matplotlib.pyplot.plot function
    fignum can be specified
    cut x data if longer than y data
    any kwargs from plot can be passed
    Use the homemade function refresh() to draw and plot the figure, no matter the way python is called (terminal, script, notebook)
    """
    xp = np.asarray(x)
    yp = np.asarray(y)
    if len(xp) > len(yp):
        print("Warning : x and y data do not have the same length")
        xp = xp[:len(yp)]

    plot(plt.plot, xp, yp, fignum=fignum, label=label, subplot=subplot, **kwargs)


def graphloglog(*args, **kwargs):
    plot(plt.loglog, *args, **kwargs)


def semilogx(*args, **kwargs):
    plot(plt.semilogx, *args, **kwargs)


def semilogy(*args, **kwargs):
    plot(plt.semilogy, *args, **kwargs)


# def errorbar(x, y, xerr, yerr, fignum=1, label='k^', subplot=None, **kwargs):
#     """
#     plot a graph using matplotlib.pyplot.errorbar function
#     fignum can be specified
#     cut x data if longer than y data
#     any kwargs from plot can be passed
#     """
#     set_fig(fignum, subplot=subplot)
#     plt.errorbar(x, y, yerr, xerr, label, **kwargs)
#     refresh()



def time_label(M, frame):
    Dt = M.t[frame + 1] - M.t[frame]
    title = 't = ' + str(floor(M.t[frame] * 1000) / 1000.) + ' s, Dt = ' + str(floor(Dt * 10 ** 4) / 10.) + ' ms'
    return title


def pause(time=3):
    plt.pause(time)


def refresh(hold=True, block=False, ipython=True):
    """
    Depreciated. Use plt.show()- Takumi 1/11/18

    Refresh the display of the current figure.
    INPUT
    -----
    hold (opt): False if the display has overwritten.
    OUTPUT
    -----
    None
    """

    plt.pause(10.0)
    plt.draw()

    if not ipython:
        plt.hold(hold)
        plt.show(block)


def subplot(i, j, k):
    plt.subplot(i, j, k)


def legend_stephane(x_legend, y_legend, title, display=False, cplot=False, show=False, fontsize=__fontsize__):
    """
    Add a legend to the current figure
        Contains standard used font and sizes
        return a default name for the figure, based on x and y legends

    Parameters
    ----------
    x_legend : str
        x label
    y_legend : str
        y label
    title : str
        title label
    colorplot : bool
        default False
        if True, use the title for the output legend

    Returns
    -------
    fig : dict
        one element dictionnary with key the current figure number
        contain a default filename generated from the labels
    """
    # additional options ?
    plt.rc('font', family='Times New Roman')
    plt.xlabel(x_legend, fontsize=fontsize)
    plt.ylabel(y_legend, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)

    if show:
        refresh()

    # fig is a dictionary where the key correspond to the fig number and the element to the automatic title
    fig = figure_label(x_legend, y_legend, title, display=display, cplot=cplot)
    fig = get_data(fig)

    return fig


def get_data(fig, cplot=False):
    """

    fig :
    cplot :
    """
    current = plt.gcf()
    lines = plt.gca().get_lines()

    Dict = {}
    for i, line in enumerate(lines):
        xd = line.get_xdata()
        yd = line.get_ydata()
        Dict['xdata_' + str(i)] = xd
        Dict['ydata_' + str(i)] = yd

        if cplot:
            zd = line.get_zdata()
            Dict['zdata' + str(i)] = zd

    fig[current.number]['data'] = Dict
    return fig


def figure_label(x_legend, y_legend, title, display=True, cplot=False, include_title=False):
    """

    Parameters
    ----------
    x_legend :
    y_legend :
    title :
    display :
    cplot :
    include_title :

    Returns
    -------
    """
    # generate a standard name based on x and y legend, to be used by default as a file name output
    x_legend = remove_special_chars(x_legend)
    y_legend = remove_special_chars(y_legend)
    title = remove_special_chars(title)

    fig = {}
    current = plt.gcf()
    fig[current.number] = {}
    if cplot:
        fig[current.number]['fignum'] = title  # +'_'+x_legend+'_'+y_legend #start from the plotted variable (y axis)
    else:
        fig[current.number]['fignum'] = y_legend + '_vs_' + x_legend  # start from the plotted variable (y axis)

    if include_title:
        fig[current.number]['fignum'] = y_legend + '_vs_' + x_legend + '_' + title

    if display:
        print(current.number, fig[current.number])
    return fig


def remove_special_chars(string, chars_rm=['$', '\ ', '[', ']', '^', '/', ') ', '} ', ' '],
                         chars_rp=['{', '(', ',', '=', '.']):
    """
    Remove characters from a typical latex format to match non special character standards

    Parameters
    ----------
    string : str
        input string
    chars_rm : str list. Default value : ['$','\ ',') ']
        char list to be simply removed
    chars_rp : str list. Default value : ['( ',',']
        char list to be replaced by a '_'

    Returns
    -------
    string : str
        modified string
    """
    for char in chars_rm:
        string = string.replace(char[0], '')
    for char in chars_rp:
        string = string.replace(char[0], '_')
    return string


def titleS(M):
    """standard title format to know from what date is has been created
    Need something much more sophisticated than that !
    Create a dictionary of keyword to be added in the title, and add every element into a string formatted style
    """
    tdict = {}
    tdict['type'] = M.param.typeplane
    if M.param.typeplane == 'sv':
        tdict['X'] = M.param.Xplane
    if M.param.typeplane == 'bv':
        tdict['Z'] = M.param.Zplane

    tdict['fx'] = M.param.fx

    title = M.Id.get_id() + ','
    for key in tdict.keys():
        title += key + '=' + str(tdict[key]) + ','

    return title





def set_title(M, opt=''):
    """

    Parameters
    ----------
    M :
    opt :

    Returns
    -------
    title : str
    """
    # if Zplane attribute exist !
    title = 'Z= ' + str(int(M.param.Zplane)) + ', ' + M.param.typeview + ', mm, ' + M.Id.get_id() + ', ' + opt
    plt.title(title, fontsize=18)
    return title


def set_name(M, param=[]):
    # if Zplane attribute exist !
    s = ''
    for p in param:
        try:
            pi = int(getattr(M.param, p))
        except:
            pi = getattr(M.param, p)
        s = s + p + str(pi) + '_'
    # s =s[:-1]
    # plt.title(title, fontsize=18)
    return s


def clegende(c, c_legend):
    """Set a label to the object c"""
    c.set_label(c_legend)


def save_graphes(M, figs, prefix='', suffix=''):
    save_figs(figs, savedir='./Results/' + os.path.basename(M.dataDir) + '/', prefix=prefix, suffix=suffix)


def save_figs(figs, savedir='./', suffix='', prefix='', frmt='pdf', dpi=300, display=False, data_save=True):
    """Save a dictionnary of labeled figures using dictionnary elements
    dict can be autogenerated from the output of the graphes.legende() function
    by default the figures are saved whithin the same folder from where the python code has been called

    Parameters
    ----------
    figs : dict of shape {int:str,...}
        the keys correspond to the figure numbers, the associated field
    savedir : str. default : './'
    frmt : str
        file Format
    dpi : int
        division per inch. set the quality of the output

    Returns
    -------
    None
    """
    c = 0
    filename = ''
    for key in figs.keys():
        fig = figs[key]
        # save the figure
        filename = savedir + prefix + fig['fignum'] + suffix + '_ag'
        print 'graphes.save_figs(): filename = ', filename
        save_fig(key, filename, frmt=frmt, dpi=dpi)
        c += 1

        if data_save:
            # save the data
            h5py_s.save(filename, fig['data'])

    if display:
        print('Number of auto-saved graphs : ' + str(c))
        print(filename)


def save_fig(fignum, filename, frmt='pdf', dpi=300, overwrite=True):
    """
    Save the figure fignumber in the given filename

    Parameters
    ----------
    fignum : int
        number of the fig to save: this int is matplotlib's figure assignment
        We should really make this an optional argument, with default==1
    filename : str
        name of the file to save
    frmt : str (optional, default='pdf')
        File format in which to save the image
    dpi : int (optional, default=300)
        number of dpi for image based format. Default is 300
    overwrite : bool
        Whether to overwrite an existing saved image with the current one

    Returns
    -------
    None
    """
    # If the path in filename is supplied as a relative path, make it an absolute path
    if os.path.dirname(filename)[0] == '.':
        savedir = os.path.dirname(filename)
        print 'graphes.save_fig(): savedir = ', savedir
        # Stephane remade the filename from itself here for some reason. This causes problems if we remove relative
        # specifier ('.') from filename.
        # filename = savedir + '/' + os.path.basename(filename)
    else:
        savedir = os.path.dirname(filename)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    filename = filename + '.' + frmt
    if fignum != 0:
        plt.figure(fignum)

    if not os.path.isfile(filename):
        plt.savefig(filename, dpi=dpi)
    else:
        if overwrite:
            plt.savefig(filename, dpi=dpi)


def plot_axes(fig, num):
    ax = fig.add_subplot(num)
    ax.set_aspect('equal', adjustable='box')
    # graphes.legende('','','Front view')
    # draw_fieldofview(M.Sdata,ax3,view='front')
    return ax





def get_axis_coord(M, direction='v'):
    X = M.x
    Y = M.y

    if hasattr(M, 'param'):
        if M.param.angle == 90:
            Xb = X
            Yb = Y
            X = Yb
            Y = Xb
    # return rotate(X,Y,M.param.angle)
    return X, Y


def rotate(X, Y, angle):
    angle = angle / 180 * np.pi
    return X * np.cos(angle) - Y * np.sin(angle), Y * np.cos(angle) + X * np.sin(angle)


def Mplot(M, field, frame, auto_axis=False, step=1, W=None, Dt=None, fignum=1, show=False, vmin=0, vmax=0, log=False,
          display=False, tstamp=False, compute=False, cbar=False, colorbar=False):
    """

    Parameters
    ----------
    M :
    field :
    frame :
    auto_axis :
    step :
    W :
    Dt :
    fignum :
    show :
    vmin :
    vmax :
    log :
    display :
    tstamp :
    compute :
    cbar :
    colorbar :

    Returns
    -------
    """
    import library.pprocess.check_piv as check
    import library.manager.access as access

    data = access.get(M, field, frame, step=1, compute=compute)
    dimensions = data.shape

    if field == 'strain':
        # tensor variable. chose the trace (2d divergence !):
        data = data[..., 1, 1, :] + data[..., 0, 0, :]
        # print(data)

    X, Y = get_axis_coord(M)
    jmin = 0
    data = data[:, jmin:]
    X = X[:, jmin:]
    Y = Y[:, jmin:]

    t = M.t[frame]
    ft = M.t[frame + 1] - M.t[frame]
    dx = np.mean(np.diff(M.x[0, :]))

    if dx == 0:
        dx = 1

    if vmin == 0 and vmax == 0:
        if auto_axis:
            std = np.sqrt(np.nanmedian(np.power(data, 2)))
            vmax = 10 * std
            vmin = -vmax
            if field in ['E', 'enstrophy']:
                vmin = 0

            else:
                if W is None:
                    vmin, vmax = check.bounds(M, t0=frame)
                else:
                    vmin, vmax = check.bounds_pix(W)

            if Dt is not None:
                data = data / Dt

            if field in ['Ux', 'Uy']:
                vmax = np.abs(vmax)
                vmin = -np.abs(vmax)  # *100

            if field in ['omega']:
                data = data
                vmax = np.abs(vmax) / 5.  # *15#/5.
                vmin = -np.abs(vmax)  # *10#*100#vmax

            if field in ['strain']:
                data = data
                vmax = np.abs(vmax) / 20.  # *15#/5.
                vmin = -np.abs(vmax)  # *10#*100#vmax

            if field in ['E']:
                # std = np.std(data[...,frame])
                vmax = vmax ** 2
                vmin = vmin ** 2

            if field in ['enstrophy']:
                vmax = (vmax / 5.) ** 2
                vmin = (vmin) ** 2

    if log:
        vmax = np.log10(vmax)
        if vmin > 0:
            vmin = np.log10(vmin)
        else:
            vmin = vmax / 100.
    n = (X.shape[0] - dimensions[0]) / 2
    if n != 0:
        X = X[n:-n, n:-n]
        Y = Y[n:-n, n:-n]
    color_plot(X, Y, data[..., 0], show=show, fignum=fignum, vmin=vmin, vmax=vmax, log10=False, cbar=cbar)
    #    time_stamp(M,frame)
    if colorbar == True:
        plt.colorbar()

    # plt.axis('equal')
    if tstamp:
        t = M.t[frame]
        Dt = M.t[frame + 1] - M.t[frame]
        s = ', t = ' + str(np.round(t * 1000) / 1000) + ' s, Dt = ' + str(np.round(Dt * 10000) / 10) + 'ms'
    else:
        s = ''

    figs = {}
    figs.update(legend('X (mm)', 'Y (mm)', field + s, display=display, cplot=True, show=show))

    return figs


#    title = os.path.basename(M.Sdata.fileCine)

def movie(M, field, indices=None, compute=False, label='', Dirname='./', tracking=False, **kwargs):
    """
    Generates png files of heatmap of specified field in specified directory

    Parameters
    ----------
    M : M class object
    field : E, omega, enstrophy
    indices : tuple
              e.g. indices=range(500,1000) saves the image files of the specified heatmap between 500-th and 999-th frame
    compute :
    label :
    Dirname : string
              name of the directory where the image files will be stored
    tracking :
    kwargs : keys are vmin, vmax, and possibly more.

    Returns
    -------
    """
    figs = {}
    if indices == None:
        nx, ny, nt = M.shape()
        indices = range(1, nt - 1)
    if tracking:
        import turbulence.vortex.track as track

    Dirname = Dirname + 'Movie_' + field + M.Id.get_id() + '/'

    fignum = 1
    fig, ax = set_fig(fignum, subplot=111)
    plt.clf()

    start = True
    for frame in indices:
        figs.update(Mplot(M, field, frame, compute=compute, **kwargs))

        if tracking:
            tup = track.positions(M, frame, field='omega', display=False, sigma=3., fignum=fignum)
            # print(tup)
            graph([tup[0]], [tup[2]], label='ro', linewidth=3, fignum=fignum)
            graph([tup[1]], [tup[3]], label='bo', linewidth=3, fignum=fignum)

        if start:
            colorbar(label=label)
            start = False

        # print(Dirname)
        save_figs(figs, savedir=Dirname, suffix='_' + str(frame), dpi=100, frmt='png', data_save=False)

        plt.cla()


def time_stamp(M, ii, x0=-80, y0=50, fignum=None):
    """
    Parameters
    ----------
    M :
    ii :
    x0 :
    y0 :
    fignum :
    """
    t = M.t[ii]
    Dt = M.t[ii + 1] - M.t[ii]
    s = 't = ' + str(np.round(t * 1000) / 1000) + ' s, Dt = ' + str(np.round(Dt * 10000) / 10) + 'ms'
    #   print(s)
    if fignum is not None:
        set_fig(fignum)
    plt.text(x0, y0, s, fontsize=20)


def vfield_plot(M, frame, fignum=1):
    """
    Plot a 2d velocity fields with color coded vectors
    Requires fields for the object M : Ux and Uy
    INPUT
    -----
    M : Mdata set of measure
    frame : number of the frame to be analyzed
    fignum (opt) : asking for where the figure should be plotted
    OUTPUT
    ------
    None
    	"""
    x = M.x
    y = M.y
    Ux = M.Ux[:, :, frame]
    Uy = M.Uy[:, :, frame]

    colorCodeVectors = True
    refVector = 1.
    vectorScale = 100
    vectorColormap = 'jet'

    # bounds
    # chose bounds from the histograme of E values ?
    scalarMinValue = 0
    scalarMaxValue = 100

    # make the right figure active
    set_fig(fignum)

    # get axis handle
    ax = plt.gca()
    ax.set_yticks([])
    ax.set_xticks([])

    E = np.sqrt(Ux ** 2 + Uy ** 2)
    Emoy = np.nanmean(E)

    if colorCodeVectors:
        Q = ax.quiver(x, y, Ux / Emoy, Uy / Emoy, E, \
                      scale=vectorScale / refVector,
                      scale_units='width',
                      cmap=plt.get_cmap(vectorColormap),
                      clim=(scalarMinValue, scalarMaxValue),
                      edgecolors=('none'),
                      zorder=4)
        # elif settings.vectorColorValidation:
        #    v = 1
        #    #ax.quiver(x[v==0], y[v==0], ux[v==0], uy[v==0], \
        #    scale=vectorScale/refVector, scale_units='width', color=[0, 1, 0],zorder=4)
    #    Q = ax.quiver(x[v==1], y[v==1], ux[v==1], uy[v==1], \
    #                  scale=vectorScale/refVector, scale_units='width', color='red',zorder=4)
    else:
        Q = ax.quiver(x, y, Ux / E, Uy / E, scale=vectorScale / refVector, scale_units='width',
                      zorder=4)  # , color=settings.vectorColor

    legend('$x$ (mm)', '$y$ (mm)', '')

    # add reference vector
    # if settings.showReferenceVector:
    #        plt.quiverkey(Q, 0.05, 1.05, refVector, str(refVector) + ' m/s', color=settings.vectorColor)

    # overwrite existing colorplot
    refresh(False)


######################################################################
#################### Histograms and pdfs #############################
######################################################################

def vplot(M):
    pass


def hist(Y, Nvec=1, fignum=1, num=100, step=None, label='o-', log=False, normalize=True, xfactor=1, **kwargs):
    """
    Plot histogramm of Y values
    """
    set_fig(fignum)
    # print('Number of elements :'+str(len(Y)))
    if step is None:
        n, bins = np.histogram(np.asarray(Y), bins=num, **kwargs)
        #  print(bins)
    else:
        d = len(np.shape(Y))
        #        print('Dimension : '+str(d))
        N = np.prod(np.shape(Y))
        if N < step:
            step = N
        n, bins = np.histogram(np.asarray(Y), bins=int(N / step))

    if normalize:
        dx = np.mean(np.diff(bins))
        n = n / (np.sum(n) * dx)

    xbin = (bins[0:-1] + bins[1:]) / 2 / xfactor
    n = n * xfactor

    if log:
        # Plot in semilogy plot
        semilogy(xbin / Nvec, n, fignum, label)
    else:
        plt.plot(xbin, n, label)
        plt.axis([np.min(xbin), np.max(xbin), 0, np.max(n) * 1.1])

    refresh()
    return xbin, n


def pdf(M, field, frame, Dt=10, Dx=1024, label='ko-', fignum=1, a=15., norm=True, sign=1):
    import library.manager.access as access
    Up = access.get(M, field, frame, Dt=Dt)

    limits = [(0, Dx), (0, Dx)]
    Up = sign * access.get_cut(M, field, limits, frame, Dt=Dt)

    figs = distribution(Up, normfactor=1, a=a, label=label, fignum=fignum, norm=norm)

    return figs


def pdf_ensemble(Mlist, field, frame, Dt=10, Dx=1024, label='r-', fignum=1, a=10., norm=True, model=False):
    import library.manager.access as access

    U_tot = []

    for M in Mlist:
        pdf(M, field, frame, Dt=Dt, Dx=Dx, label='k', fignum=fignum, a=a, norm=False)

        Up = access.get(M, field, frame, Dt=Dt)
        # limits = [(0,Dx),(0,Dx)]
        #    Up = access.get_cut(M,field,limits,frame,Dt=Dt)
        # if Dx is larger than the box size, just keep all the data
        U_tot = U_tot + np.ndarray.tolist(Up)

    N = len(Mlist)
    U_tot = np.asarray(U_tot)

    x, y, figs = distribution(U_tot, normfactor=N, a=a, label=label, fignum=fignum, norm=norm)

    if model:
        n = len(y)
        b = y[n // 2]
        Dy = np.log((y[n // 2 + n // 8] + y[n // 2 - n // 8]) / 2. / b)

        a = - Dy / x[n // 2 + n // 8] ** 2

        P = b * np.exp(-a * x ** 2)
        semilogy(x, P, label='b.-', fignum=fignum)

    set_axis(min(x), max(x), 1, max(y) * 2)
    if field == 'omega' or field == 'strain':
        unit = ' (s^-1)'
    elif field == 'E':
        unit = 'mm^2/s^2'
    else:
        unit = ' (mm/s)'
    figs = {}
    figs.update(legend(field + unit, field + ' PDF', time_label(M, frame)))
    return figs


def avg_from_dict(dd, keyx, keyy, times, fignum=1, display=True, label='b-'):
    """
    Compute the average function from a dictionnary with keys (time,keyx) (time,keyy) for time in times

    """
    avg = {}
    avg[keyx] = np.mean([dd[(time, keyx)] for time in times], axis=0)
    avg[keyy] = np.mean([dd[(time, keyy)] for time in times], axis=0)

    std = {}
    std[keyx] = np.std([dd[(time, keyx)] for time in times], axis=0)
    std[keyy] = np.std([dd[(time, keyy)] for time in times], axis=0)

    if display:
        for time in times:
            graph(dd[(time, keyx)], dd[(time, keyy)], label='k-', fignum=fignum, color='0.7')

        errorbar(avg[keyx], avg[keyy], std[keyx], std[keyy], fignum=fignum, label=label)

    return avg, std


def distribution(Y, normfactor=1, a=10., label='k', fignum=1, norm=True):
    Y = np.asarray(Y)
    Median = np.sqrt(np.nanmedian(Y ** 2))

    "test if the field is positive definite"
    t = Y >= 0

    if norm:
        bound = a
    else:
        bound = a * Median
    step = bound / 10 ** 2.5

    if t.all():
        x = np.arange(0, bound, step)
    else:
        x = np.arange(-bound, bound, step)

    if norm:
        Y = Y / Median

    n, bins = np.histogram(Y, bins=x)
    xbin = (bins[:-1] + bins[1:]) / 2
    n = n / normfactor  # in case of several series (ensemble average)

    semilogy(xbin, n, label=label, fignum=fignum)
    set_axis(min(xbin), max(xbin), min(n) / 2, max(n) * 2)
    figs = {}
    figs.update(legend('', 'PDF', ''))

    val = 0.5
    x_center = xbin[np.abs(xbin) < val]
    n_center = n[np.abs(xbin) < val]

    moy = np.sum(n * xbin) / np.sum(n)
    std = np.sum(n * (xbin - moy) ** 2) / np.sum(n)

    print("Distribution : " + str(moy) + ' +/- ' + str(std))
    #    a = fitting.fit(fitting.parabola,x_center,n_center)
    #    n_th = fitting.parabola(xbin,a)
    #    graph(xbin,n_th,label='r-',fignum=fignum)
    return xbin, n, figs
    ####### to add :

    #   1. streamline plots
    #   2. Strain maps
    #   3. Vorticity maps

    # for i in range(10,5000,1):
    #    vfield_plot(M_log[4],i,1)
    # input()
