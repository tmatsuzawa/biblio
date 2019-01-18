import numpy as np
from scipy.optimize import curve_fit
import library.basics.std_func as std_func

'''
Module for plotting and saving figures
'''


import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pylab as pylab
import matplotlib.ticker as ticker
import mpl_toolkits.axes_grid as axes_grid
import itertools
from scipy import stats
import library.basics.formatarray as fa
import numpy as np
import glob


#Global variables
#Default color cycle: iterator which gets repeated if all elements were exhausted
#__color_cycle__ = itertools.cycle(iter(plt.rcParams['axes.prop_cycle'].by_key()['color']))
__def_colors__ = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
__color_cycle__ = itertools.cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])  #matplotliv v2.0
__old_color_cycle__ = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])  #matplotliv classic
__fontsize__ = 16
__figsize__ = (8, 8)

# See all available arguments in matplotlibrc
params = {'figure.figsize': __figsize__,
          'font.size': __fontsize__,  #text
        'legend.fontsize': 12, # legend
         'axes.labelsize': __fontsize__, # axes
         'axes.titlesize': __fontsize__,
         'xtick.labelsize': __fontsize__, # tick
         'ytick.labelsize': __fontsize__}


## Save a figure
def save(path, ext='pdf', close=False, verbose=True, fignum=None, dpi=None, overwrite=True, **kwargs):
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
        fig = plt.gcf()
    else:
        fig = plt.figure(fignum)
    if dpi is None:
        dpi = fig.dpi

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
    # if a figure already exists AND you'd like to overwrite, name a figure differently
    ver_no = 0
    while os.path.exists(savepath) and not overwrite:
        savepath = directory + os.path.split(path)[1] + '_{n:03d.}'.format(n=ver_no) + ext
        ver_no += 1


    if verbose:
        print("Saving figure to '%s'..." % savepath),

    # Save the figure
    plt.savefig(savepath, dpi=dpi, **kwargs)

    # Close it
    if close:
        plt.close()

    if verbose:
        print("... Done")


## Create a figure and axes
def set_fig(fignum, subplot=None, dpi=100, figsize=None, **kwargs):
    """
    Make a plt.figure instance and makes an axes as an attribute of the figure instance
    Returns figure and ax
    Parameters
    ----------
    fignum
    subplot
    dpi
    figsize
    kwargs

    Returns
    -------
    fig
    ax

    """

    if fignum == -1:
        if figsize is not None:
            fig = plt.figure(dpi=dpi, figsize=figsize)
        else:
            fig = plt.figure(dpi=dpi)
    if fignum == 0:
        fig = plt.cla()  #clear axis
    if fignum > 0:
        if figsize is not None:
            fig = plt.figure(num=fignum, dpi=dpi, figsize=figsize)
            fig.set_size_inches(figsize[0], figsize[1])
        else:
            fig = plt.figure(num=fignum, dpi=dpi)
        fig.set_dpi(dpi)

    if subplot is not None:
        # a triplet is expected !
        ax = fig.add_subplot(subplot, **kwargs)
        return fig, ax
    else:
        ax = fig.add_subplot(111)
        return fig, ax


def plotfunc(func, x, param, fignum=1, subplot=111, ax = None, label=None, color=None, linestyle='-', legend=False, figsize=__figsize__, **kwargs):
    """
    plot a graph using the function fun
    fignum can be specified
    any kwargs from plot can be passed
    Use the homemade function refresh() to draw and plot the figure, no matter the way python is called (terminal, script, notebook)
    """
    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = plt.gcf()


    # y = func(x, a, b)
    if len(param)==1:
        a=param[0]
        y = func(x, a)
    if len(param) == 2:
        a, b = param[0], param[1]
        y = func(x, a, b)
    if len(param) == 3:
        a, b, c = param[0], param[1], param[2]
        y = func(x, a, b, c)
    if len(param) == 4:
        a, b, c, d = param[0], param[1], param[2], param[3]
        y = func(x, a, b, c, d)
    if not color==None:
        ax.plot(x, y, color=color, linestyle=linestyle, label=label, **kwargs)
    else:
        ax.plot(x, y, label=label, linestyle=linestyle, **kwargs)
    if legend:
        ax.legend()
    return fig, ax

def plot(x, y, fignum=1, figsize=None, label='', color=None, subplot=None, legend=False, **kwargs):
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
        plt.plot(x, y, color=color, label=label, **kwargs)
    if legend:
        plt.legend()
    return fig, ax


def scatter(x, y, ax=None, fignum=1, figsize=None, marker='o', fillstyle='full', label=None, subplot=None, legend=False, **kwargs):
    """
    plot a graph using given x,y
    fignum can be specified
    any kwargs from plot can be passed
    Use the homemade function refresh() to draw and plot the figure, no matter the way python is called (terminal, script, notebook)
    """
    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = ax.get_figure()
    x, y = np.array(x), np.array(y)
    if len(x.flatten()) > len(y.flatten()):
        print("Warning : x and y data do not have the same length")
        x = x[:len(y)]
    if fillstyle =='none':
        # Scatter plot with open markers
        facecolors = 'none'
        # ax.scatter(x, y, color=color, label=label, marker=marker, facecolors=facecolors, edgecolors=edgecolors, **kwargs)
        ax.scatter(x, y, label=label, marker=marker, facecolors=facecolors, **kwargs)
    else:
        ax.scatter(x, y, label=label, marker=marker, **kwargs)
    if legend:
        plt.legend()
    return fig, ax


def pdf(data, nbins=10, return_data=False, vmax=None, vmin=None, fignum=1, figsize=None, subplot=None, density=True, analyze=False, **kwargs):
    def compute_pdf(data, nbins=10):
        # Get a normalized histogram
        # exclude nans from statistics
        hist, bins = np.histogram(data.flatten()[~np.isnan(data.flatten())], bins=nbins, density=density)
        # len(bins) = len(hist) + 1
        # Get middle points for plotting sake.
        bins1 = np.roll(bins, 1)
        bins = (bins1 + bins) / 2.
        bins = np.delete(bins, 0)
        return bins, hist

    data = np.asarray(data)

    # Use data where values are between vmin and vmax
    if vmax is not None:
        cond1 = np.asarray(data) < vmax # if nan exists in data, the condition always gives False for that data point
    else:
        cond1 = np.ones(data.shape, dtype=bool)
    if vmin is not None:
        cond2 = np.asarray(data) > vmin
    else:
        cond2 = np.ones(data.shape, dtype=bool)
    data = data[cond1 * cond2]

    # compute a pdf
    bins, hist = compute_pdf(data, nbins=nbins)
    fig, ax = plot(bins, hist, fignum=fignum, figsize=figsize, subplot=subplot, **kwargs)

    if analyze:
        bin_width = float(bins[1]-bins[0])
        mean = np.nansum(bins * hist * bin_width)
        mode = bins[np.argmax(hist)]
        var = np.nansum(bins**2 * hist * bin_width)
        text2 = 'mean: %.2f' % mean
        text1 = 'mode: %.2f' % mode
        text3 = 'variance: %.2f' % var
        addtext(ax, text=text2, option='tc2')
        addtext(ax, text=text1, option='tc')
        addtext(ax, text=text3, option='tc3')

    if not return_data:
        return fig, ax
    else:
        return fig, ax, bins, hist


def errorbar(x, y, xerr=0, yerr=0, fignum=1, marker='o', fillstyle='full', linestyle='None', label=None, mfc='white', subplot=None, legend=False, figsize=None, **kwargs):
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
    fig, ax = set_fig(fignum, subplot, figsize=figsize)
    # Make sure that xerr and yerr are numpy arrays
    ## x, y, xerr, yerr do not have to be numpy arrays. It is just a convention. - takumi 04/01/2018
    x, y = np.array(x), np.array(y)
    # Make xerr and yerr numpy arrays if they are not scalar. Without this, TypeError would be raised.
    if not (isinstance(xerr, int) or isinstance(xerr, float)):
        xerr = np.array(xerr)
    if not (isinstance(yerr, int) or isinstance(yerr, float)):
        yerr = np.array(yerr)

    if fillstyle == 'none':
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, marker=marker, mfc=mfc, linestyle=linestyle, label=label, **kwargs)
    else:
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, marker=marker, fillstyle=fillstyle, linestyle=linestyle, label=label, **kwargs)
    if legend:
        plt.legend()
    return fig, ax

def errorfill(x, y, yerr, fignum=1, color=None, subplot=None, alpha_fill=0.3, ax=None, label=None,
              legend=False, figsize=None, color_cycle=__color_cycle__, **kwargs):
    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = plt.gcf()

    x = np.array(x)
    y = np.array(y)

    #ax = ax if ax is not None else plt.gca()
    # if color is None:
    #     color = color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        yerrdown, yerrup = yerr
        ymin = y - yerrdown
        ymax = y + yerrup
    else:
        ymin = y - yerr
        ymax = y + yerr


    if color is not None:
        ax.plot(x, y, color=color, label=label, **kwargs)
        ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)
    else:
        ax.plot(x, y,label=label, **kwargs)
        ax.fill_between(x, ymax, ymin, alpha=alpha_fill)

    #patch used for legend
    color_patch = mpatches.Patch(color=color, label=label)
    if legend:
        plt.legend(handles=[color_patch])


    return fig, ax, color_patch


## Plot a fit curve
def plot_fit_curve(xdata, ydata, func=None, fignum=1, subplot=111, figsize=None, linestyle='--',
                   xmin=None, xmax=None, add_equation=True, eq_loc='bl', color=None, label='fit',
                   show_r2=False, **kwargs):
    """
    Plots a fit curve given xdata and ydata
    Parameters
    ----------
    xdata
    ydata
    func : Method, assumes a function to be passed
    fignum
    subplot

    Returns
    -------
    fig, ax
    popt, pcov : fit results, covariance matrix
    """

    xdata = np.array(xdata)
    ydata = np.array(ydata)

    if any(np.isnan(ydata)) or any(np.isnan(xdata)):
        print 'Original data contains np.nans! Delete them for curve fitting'
        condx, condy = np.isnan(xdata), np.isnan(ydata)
        cond = (~condx * ~condy)
        print 'No of deleted data points %d / %d' % (np.sum(~cond), len(xdata))
        if np.sum(~cond) == len(xdata):
            print 'No data points for fitting!'
            raise RuntimeError
        xdata, ydata = xdata[cond], ydata[cond]

    if xmin is None:
        xmin = np.min(xdata)
    if xmax is None:
        xmax = np.max(xdata)

    x_for_plot = np.linspace(xmin, xmax, 1000)
    if func is None or func=='linear':
        print 'Fitting to a linear function...'
        popt, pcov = curve_fit(std_func.linear_func, xdata, ydata)
        if color is None:
            fig, ax = plot(x_for_plot, std_func.linear_func(x_for_plot, *popt), fignum=fignum, subplot=subplot,
                           label=label, figsize=figsize, linestyle=linestyle)
        else:
            fig, ax = plot(x_for_plot, std_func.linear_func(x_for_plot, *popt), fignum=fignum, subplot=subplot,
                           label=label, figsize=figsize, color=color, linestyle=linestyle, **kwargs)

        if add_equation:
            text = '$y=ax+b$: a=%.2f, b=%.2f' % (popt[0], popt[1])
            addtext(ax, text, option=eq_loc)
        y_fit = std_func.linear_func(xdata, *popt)
    elif func=='power':
        print 'Fitting to a power law..'
        popt, pcov = curve_fit(std_func.power_func, xdata, ydata)
        if color is None:
            fig, ax = plot(x_for_plot, std_func.power_func(x_for_plot, *popt), fignum=fignum, subplot=subplot,
                           label=label, figsize=figsize, linestyle=linestyle, **kwargs)
        else:
            fig, ax = plot(x_for_plot, std_func.power_func(x_for_plot, *popt), fignum=fignum, subplot=subplot,
                           label=label, figsize=figsize, color=color, linestyle=linestyle, **kwargs)

        if add_equation:
            text = '$y=ax^b$: a=%.2f, b=%.2f' % (popt[0], popt[1])
            addtext(ax, text, option=eq_loc)
        y_fit = std_func.power_func(xdata, *popt)
    else:
        popt, pcov = curve_fit(func, xdata, ydata)
        if color is None:
            fig, ax = plot(x_for_plot, func(x_for_plot, *popt), fignum=fignum, subplot=subplot, label=label, figsize=figsize,
                           linestyle=linestyle, **kwargs)
        else:
            fig, ax = plot(x_for_plot, func(x_for_plot, *popt), fignum=fignum, subplot=subplot, label=label, figsize=figsize,
                           color=color, linestyle=linestyle, **kwargs)
        y_fit = func(xdata, *popt)
    #plot(x_for_plot, std_func.power_func(x_for_plot, *popt))

    if show_r2:
        # compute R^2
        # residual sum of squares
        ss_res = np.sum((ydata - y_fit) ** 2)
        # total sum of squares
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        # r-squared
        r2 = 1 - (ss_res / ss_tot)
        addtext(ax, '$R^2: %.2f$' % r2, option='bl3')



    return fig, ax, popt, pcov


## 2D plotsFor the plot you showed at group meeting of lambda converging with resolution, can you please make a version with two x axes (one at the top, one below) one pixel spacing, other PIV pixel spacing, and add a special tick on each for the highest resolution point.
# (pcolormesh)
def color_plot(x, y, z, subplot=None, fignum=1, figsize=None, vmin=None, vmax=None, log10=False, show=False,
               cbar=False, cmap='magma', aspect='equal', linewidth=0,  **kwargs):
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
    fig, ax = set_fig(fignum, subplot, figsize=figsize)
    # fig, ax = set_fig(fignum, subplot, figsize=figsize, aspect=aspect)

    if log10:
        z = np.log10(z)

    # Note that the cc returned is a matplotlib.collections.QuadMesh
    # print('np.shape(z) = ' + str(np.shape(z)))
    if vmin is None and vmax is None:
        # plt.pcolormesh returns a QuadMesh class object.
        cc = plt.pcolormesh(x, y, z, cmap=cmap, **kwargs)
    else:
        cc = plt.pcolormesh(x, y, z, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    if cbar:
        plt.colorbar()

    if aspect == 'equal':
        ax.set_aspect('equal')
    # set edge color to face color
    cc.set_edgecolor('face')

    return fig, ax, cc

#imshow
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



## Miscellanies
def show():
    plt.show()

## Lines
def axhline(ax, y, x0=None, x1=None, color='black', linestyle='--', **kwargs):
    """
    Draw a horizontal line at y=y from xmin to xmax
    Parameters
    ----------
    y
    x

    Returns
    -------

    """
    if x0 is not None:
        xmin, xmax = ax.get_xlim()
        xmin_frac, xmax_frac = x0 / float(xmax), x1 / float(xmax)
    else:
        xmin_frac, xmax_frac= 0, 1
    ax.axhline(y, xmin_frac, xmax_frac, color=color, linestyle=linestyle, **kwargs)

def axvline(ax, x, y0=None, y1=None,  color='black', linestyle='--', **kwargs):
    """
    Draw a vertical line at x=x from ymin to ymax
    Parameters
    ----------
    x
    y

    Returns
    -------

    """
    if y0 is not None:
        ymin, ymax = ax.get_ylim()
        ymin_frac, ymax_frac = y0 / float(ymax), y1 / float(ymax)
    else:
        ymin_frac, ymax_frac= 0, 1
    ax.axvline(x, ymin_frac, ymax_frac, color=color, linestyle=linestyle, **kwargs)

## Bands
def axhband(ax, y0, y1, x0=None, x1=None, color='C1', alpha=0.2, **kwargs):
    """
        Make a horizontal band between y0 and y1 (highlighting effect)
        Parameters
        ----------
        ax: plt.axes.axes object
        x0: x-coordinate of the left of a band  (x0 < x1). As a default, x0, x1 = ax.get_xlim()
        x1: x-coordinate of the right of a band (x0 < x1)
        y0: y-coordinate of the bottom of a band  (y0 < y1)
        y1: y-coordinate of the top of a band  (y0 < y1)
        color: color of a band
        alpha: alpha of a band
        kwargs: kwargs for ax.fill_between()

        Returns
        -------

        """
    if x0 is None and x1 is None:
        x0, x1 = ax.get_xlim()
    ax.fill_between(np.arange(x0, x1), y0, y1, alpha=alpha, color=color, **kwargs)

def axvband(ax, x0, x1, y0=None, y1=None, color='C1', alpha=0.2, **kwargs):
    """
    Make a vertical band between x0 and x1 (highlighting effect)
    Parameters
    ----------
    ax: plt.axes.axes object
    x0: x-coordinate of the left of a band  (x0 < x1)
    x1: x-coordinate of the right of a band (x0 < x1)
    y0: y-coordinate of the bottom of a band  (y0 < y1)
    y1: y-coordinate of the top of a band  (y0 < y1). As a default, y0, y1 = ax.get_ylim()
    color: color of a band
    alpha: alpha of a band
    kwargs: kwargs for ax.fill_between()

    Returns
    -------

    """
    xmin, xmax = ax.get_xlim()
    if y0 is None and y1 is None:
        y0, y1 = ax.get_ylim()
    ax.fill_between(np.arange(x0, x1), y0, y1, alpha=alpha, color=color, **kwargs)
    ax.set_ylim(y0, y1)
    ax.set_xlim(xmin, xmax)

## Legend
# Legend
def legend(ax, remove=False, **kwargs):
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
    leg = ax.legend(**kwargs)
    if remove:
        leg.get_frame().set_facecolor('none')


# Colorbar
# Scientific format for Color bar- set format=sfmt to activate it
sfmt=mpl.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))

def add_colorbar_old(mappable, fig=None, ax=None, fignum=None, label=None, fontsize=__fontsize__,
                 vmin=None, vmax=None, cmap='jet', option='normal', **kwargs):
    """
    Adds a color bar (Depricated. replaced by add_colorbar)
    Parameters
    ----------
    mappable : image like QuadMesh object to which the color bar applies (NOT a plt.figure instance)
    ax : Parent axes from which space for a new colorbar axes will be stolen
    label :

    Returns
    -------
    """
    # Get a Figure instance
    if fig is None:
        fig = plt.gcf()
        if fignum is not None:
            fig = plt.figure(num=fignum)
    if ax is None:
        ax = plt.gca()

    # if vmin is not None and vmax is not None:
    #     norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # elif vmin is None and vmax is not None:
    #     print 'vmin was not provided!'
    # elif vmin is not None and vmax is None:
    #     print 'vmax was not provided!'

    # fig.colorbar makes a another ax object which colives with ax in the fig instance.
    # Therefore, cb has all attributes that ax object has!

    if option == 'scientific':
        cb = fig.colorbar(mappable, ax=ax, cmap=cmap, format=sfmt, **kwargs)
    else:
        cb = fig.colorbar(mappable, ax=ax, cmap=cmap, **kwargs)

    if not label == None:
        cb.set_label(label, fontsize=fontsize)

    return cb


def add_colorbar(mappable, fig=None, ax=None, fignum=None, location='right', label=None, fontsize=None, option='normal',
                 tight_layout=True, ticklabelsize=None, aspect='equal', **kwargs):
    """
    Adds a color bar

    e.g.
        fig = plt.figure()
        img = fig.add_subplot(111)
        ax = img.imshow(im_data)
        colorbar(ax)
    Parameters
    ----------
    mappable
    location

    Returns
    -------

    """
    # ax = mappable.axes
    # fig = ax.figure
    # Get a Figure instance
    if fig is None:
        fig = plt.gcf()
        if fignum is not None:
            fig = plt.figure(num=fignum)
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()


    divider = axes_grid.make_axes_locatable(ax)
    cax = divider.append_axes(location, size='5%', pad=0.15)
    if option == 'scientific':
        cb = fig.colorbar(mappable, cax=cax, format=sfmt, **kwargs)
    else:
        cb = fig.colorbar(mappable, cax=cax, **kwargs)

    if not label is None:
        if fontsize is None:
            cb.set_label(label)
        else:
            cb.set_label(label, fontsize=fontsize)
    if ticklabelsize is not None:
        cb.ax.tick_params(labelsize=ticklabelsize)

    # Adding a color bar may distort the aspect ratio. Fix it.
    if aspect=='equal':
        ax.set_aspect('equal')

    # Adding a color bar may disport the overall balance of the figure. Fix it.
    if tight_layout:
        fig.tight_layout()

    return cb

def add_discrete_colorbar(ax, colors, vmin=0, vmax=None, label=None, fontsize=None, option='normal',
                 tight_layout=True, ticklabelsize=None, ticklabel=None,
                 aspect = None, **kwargs):
    fig = ax.get_figure()
    if vmax is None:
        vmax = len(colors)
    tick_spacing = (vmax - vmin) / float(len(colors))
    ticks = np.linspace(vmin, vmax, len(colors)+1) + tick_spacing / 2. # tick positions

    # if there are too many ticks, just use 3 ticks
    if len(ticks) > 10:
        n = len(ticks)
        ticks = [ticks[0], ticks[n/2], ticks[-2]]


    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # dummy mappable

    if option == 'scientific':
        cb = fig.colorbar(sm, ticks=ticks, format=sfmt, **kwargs)
    else:
        cb = fig.colorbar(sm, ticks=ticks, **kwargs)

    if ticklabel is not None:
        cb.ax.set_yticklabels(ticklabel)

    if not label is None:
        if fontsize is None:
            cb.set_label(label)
        else:
            cb.set_label(label, fontsize=fontsize)
    if ticklabelsize is not None:
        cb.ax.tick_params(labelsize=ticklabelsize)

    # Adding a color bar may distort the aspect ratio. Fix it.
    if aspect=='equal':
        ax.set_aspect('equal')

    # Adding a color bar may disport the overall balance of the figure. Fix it.
    if tight_layout:
        fig.tight_layout()

    return cb



def colorbar(fignum=None, label=None, fontsize=__fontsize__):
    """
    Use is DEPRECIATED. This method is replaced by add_colorbar(mappable)
    I keep this method for old codes which might have used this method
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
def labelaxes(ax, xlabel, ylabel, **kwargs):
    ax.set_xlabel(xlabel, **kwargs)
    ax.set_ylabel(ylabel, **kwargs)
# multi-color labels
def labelaxes_multicolor(ax, list_of_strings, list_of_colors, axis='x', anchorpad=0, **kwargs):
    """this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_if_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

    # x-axis label
    if axis == 'x' or axis == 'both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left', va='bottom', **kwargs))
                 for text, color in zip(list_of_strings, list_of_colors)]
        xbox = HPacker(children=boxes, align="center", pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad, frameon=False, bbox_to_anchor=(0.2, -0.09),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis == 'y' or axis == 'both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left', va='bottom', rotation=90, **kwargs))
                 for text, color in zip(list_of_strings[::-1], list_of_colors)]
        ybox = VPacker(children=boxes, align="center", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(-0.2, 0.4),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)



# Limits
def setaxes(ax, xmin, xmax, ymin, ymax):
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    return ax

## Set axes to semilog or loglog
def tosemilogx(ax=None):
    if ax == None:
        ax = plt.gca()
    ax.set_xscale("log")
def tosemilogy(ax=None):
    if ax == None:
        ax = plt.gca()
    ax.set_yscale("log")
def tologlog(ax=None):
    if ax == None:
        ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")

# Ticks
def set_xtick_interval(ax, tickint):
    """
    Sets x-tick interval as tickint
    Parameters
    ----------
    ax: Axes object
    tickint: float, tick interval

    Returns
    -------

    """
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tickint))

def set_ytick_interval(ax, tickint):
    """
    Sets y-tick interval as tickint
    Parameters
    ----------
    ax: Axes object
    tickint: float, tick interval

    Returns
    -------

    """
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tickint))



##Title
def title(ax, title, subplot=111, **kwargs):
    ax.set_title(title, **kwargs)

def suptitle(title, fignum=None, **kwargs):
    """
    Add a centered title to the figure.
    If fignum is given, it adds a title, then it reselects the figure which selected before this method was called.
    ... this is because figure class does not have a suptitle method.
    Parameters
    ----------
    title
    fignum
    kwargs

    Returns
    -------

    """
    if fignum is not None:
        plt.figure(fignum)


    plt.suptitle(title, **kwargs)




##Text
def set_standard_pos(ax):
    """
    Sets standard positions for added texts in the plot
    left: 0.025, right: 0.75
    bottom: 0.10 top: 0.90
    xcenter: 0.5 ycenter:0.5
    Parameters
    ----------
    ax

    Returns
    -------
    top, bottom, right, left, xcenter, ycenter: float, position

    """
    left_margin, right_margin, bottom_margin, top_margin = 0.025, 0.75, 0.1, 0.90

    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    width, height = np.abs(xright - xleft), np.abs(ytop - ybottom)

    if ax.get_xscale() == 'linear':
        left, right = xleft + left_margin * width, xleft + right_margin * width
        xcenter = xleft + width/2.
    if ax.get_yscale() == 'linear':
        bottom, top = ybottom + bottom_margin * height, ybottom + top_margin * height
        ycenter = ybottom + height / 2.

    if ax.get_xscale() == 'log':
        left, right = xleft + np.log10(left_margin * width), xleft + np.log10(right_margin * width)
        xcenter = xleft + np.log10(width/2.)

    if ax.get_yscale() == 'log':
        bottom, top = ybottom + np.log10(bottom_margin * height), ybottom + np.log10(top_margin * height)
        ycenter = ybottom + np.log10(height / 2.)

    return top, bottom, right, left, xcenter, ycenter, height, width


def addtext(ax, text='text goes here', x=0, y=0, color='k',
            option=None, npartition=15, **kwargs):
    """
    Adds text to a plot. You can specify the position where the texts will appear by 'option'
    | tl2    tc2    tr2 |
    | tl     tc     tr  |
    | tl3    tc3    tr3 |
    |                   |
    | cl2               |
    | cl     cc     cr  |
    | cl3               |
    |                   |
    | bl2           br2 |
    | bl      bc    br  |
    | bl3           br3 |

    Parameters
    ----------
    ax
    subplot
    text
    x
    y
    fontsize
    color
    option: default locations
    kwargs

    Returns
    ax : with a text
    -------

    """
    top, bottom, right, left, xcenter, ycenter, height, width = set_standard_pos(ax)
    dx, dy = width / npartition,  height / npartition




    if option == None:
        ax.text(x, y, text, color=color, **kwargs)
    if option == 'tr':
        ax.text(right, top, text, color=color, **kwargs)
    if option == 'tr2':
        ax.text(right, top + dy, text,  color=color, **kwargs)
    if option == 'tr3':
        ax.text(right, top - dy, text,  color=color, **kwargs)
    if option == 'tl':
        ax.text(left, top, text,  color=color, **kwargs)
    if option == 'tl2':
        ax.text(left, top + dy, text,  color=color, **kwargs)
    if option == 'tl3':
        ax.text(left, top - dy, text,  color=color, **kwargs)

    if option == 'tc':
        ax.text(xcenter, top, text,  color=color, **kwargs)
    if option == 'tc2':
        ax.text(xcenter, top + dy, text,  color=color, **kwargs)
    if option == 'tc3':
        ax.text(xcenter, top - dy, text,  color=color, **kwargs)
    if option == 'br':
        ax.text(right, bottom, text,  color=color, **kwargs)
    if option == 'br2':
        ax.text(right, bottom + dy, text,  color=color, **kwargs)
    if option == 'br3':
        ax.text(right, bottom - dy, text, color=color, **kwargs)
    if option == 'bl':
        ax.text(left, bottom, text, color=color, **kwargs)
    if option == 'bl2':
        ax.text(left, bottom + dy, text,  color=color, **kwargs)
    if option == 'bl3':
        ax.text(left, bottom - dy, text, color=color, **kwargs)
    if option == 'bc':
        ax.text(xcenter, bottom, text, color=color, **kwargs)
    if option == 'bc2':
        ax.text(xcenter, bottom + dy, text, color=color, **kwargs)
    if option == 'bc3':
        ax.text(xcenter, bottom - dy, text, color=color, **kwargs)
    if option == 'cr':
        ax.text(right, ycenter, text, color=color, **kwargs)
    if option == 'cl':
        ax.text(left, ycenter, text, color=color, **kwargs)
    if option == 'cl2':
        ax.text(left, ycenter + dy, text, color=color, **kwargs)
    if option == 'cl3':
        ax.text(left, ycenter - dy, text, color=color, **kwargs)
    if option == 'cc':
        ax.text(xcenter, ycenter, text,  color=color, **kwargs)
    return ax




##Clear plot
def clf(fignum=None):
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

def get_default_color_cycle():
    return __color_cycle__

def get_first_n_colors_from_color_cycle(n):
    color_list = []
    for i in range(n):
        color_list.append(next(__color_cycle__))
    return color_list

def get_color_list_gradient(color1='greenyellow', color2='darkgreen', n=10):
    """
    Returns a list of colors in RGB between color1 and color2
    Input (color1 and color2) can be RGB or color names set by matplotlib
    Parameters
    ----------
    color1
    color2
    n: length of the returning list

    Returns
    -------

    """
    # convert color names to rgb if rgb is not given as arguments
    if not color1[0] == '#':
        color1 = cname2hex(color1)
    if not color2[0] == '#':
        color2 = cname2hex(color2)
    color1_rgb = hex2rgb(color1) / 255.  # np array
    color2_rgb = hex2rgb(color2) / 255.  # np array

    r = np.linspace(color1_rgb[0], color2_rgb[0], n)
    g = np.linspace(color1_rgb[1], color2_rgb[1], n)
    b = np.linspace(color1_rgb[2], color2_rgb[2], n)
    color_list = zip(r, g, b)
    return color_list


def hex2rgb(hex):
    """

    Parameters
    ----------
    hex: str, hex code. e.g. #B4FBB8

    Returns
    -------
    rgb: numpy array. RGB

    """
    h = hex.lstrip('#')
    rgb = np.asarray(list(int(h[i:i + 2], 16) for i in (0, 2, 4)))
    return rgb

def cname2hex(cname):
    """

    Parameters
    ----------
    hex: str, hex code. e.g. #B4FBB8

    Returns
    -------
    rgb: numpy array. RGB

    """
    colors = dict(mpl.colors.BASE_COLORS, **mpl.colors.CSS4_COLORS) # dictionary. key: names, values: hex codes
    try:
        hex = colors[cname]
        return hex
    except NameError:
        print cname, ' is not registered as default colors by matplotlib!'
        return None

def set_color_cycle(ax, colors=__def_colors__):
    """
    Sets a color cycle using a list
    Parameters
    ----------
    ax
    colors: list of colors in rgb/cnames/hex codes

    Returns
    -------

    """
    ax.set_prop_cycle(color=colors)

def set_color_cycle_gradient(ax, color1='greenyellow', color2='navy', n=10):
    colors = get_color_list_gradient(color1, color2, n=n)
    ax.set_prop_cycle(color=colors)



# Figure settings
def update_figure_params(params):
    """
    update a default matplotlib setting
    e.g. params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
    Parameters
    ----------
    params: dictionary

    Returns
    -------

    """
    pylab.rcParams.update(params)

def reset_figure_params():
    pylab.rcParams.update(params)

def default_figure_params():
    mpl.rcParams.update(mpl.rcParamsDefault)

# Use the settings above as a default
reset_figure_params()


# Embedded plots
def add_subplot_axes(ax, rect, axisbg='w'):
    """
    Creates a sub-subplot inside the subplot (ax)
    Parameters
    ----------
    ax
    rect: list, [x, y, width, height]  e.g. rect = [0.2,0.2,0.7,0.7]
    axisbg: background color of the newly created axes object

    Returns
    -------
    subax, Axes class object

    """
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x, y, width,height], axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax



######################
######################
######################
######################
######################
######################

# def plot(fun, x, y, fignum=1, label=None, subplot=None, **kwargs):
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

def graph(x, y, fignum=1, label=None, subplot=None, **kwargs):
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


def pdf_s(M, field, frame, Dt=10, Dx=1024, label='ko-', fignum=1, a=15., norm=True, sign=1):
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

