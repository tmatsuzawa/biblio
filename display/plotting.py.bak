import weakref
import turbulence.display.colormaps as lecmaps
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm
import turbulence.display.science_plot_style as sps
import sys

"""General functions for plotting"""


def plot_pcolormesh_scalar(x, y, C, outpath, title, xlabel=None, ylabel=None, title2='', subtext='', subsubtext='',
                           vmin='auto', vmax='auto', cmap="coolwarm", show=False, close=True, axis_on=True, FSFS=20):
    """Save a single-panel plot of a scalar quantity C as colored pcolormesh

    Parameters
    ----------
    x, y : NxN mesh arrays
        the x and y positions of the points evaluated to Cx, Cy
    C : NxN arrays
        values for the plotted quantity C evaluated at points (x,y)
    outpath : string
        full name with file path
    title : string
        title of the plot
    title2 : string
        placed below title
    subtext : string
        placed below plot
    subsubtext : string
        placed at bottom of image
    vmin, vmax : float
        minimum, maximum value of C for colorbar; default is range of values in C
    cmap : matplotlib colormap
    show : bool
        whether to display the plot for interactive viewing
    close : bool
        whether to close the plot at end of function
    axis_on : bool
        if False, axis labels will be removed
    """
    if (cmap == 'coolwarm' or cmap == 'seismic') and (vmin == 'auto' and vmax == 'auto'):
        # symmetric colormaps call for symmetric limits
        vmax = np.max(np.abs(C.ravel()))
        vmin = - vmax
    if isinstance(vmin, str):
        vmin = np.nanmin(C)
    if isinstance(vmax, str):
        vmax = np.nanmax(C)

    plt.close('all')
    fig, ax = plt.subplots(1, 1)
    # scatter scale (for color scale)
    scsc = ax.pcolormesh(x, y, C, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')
    if not axis_on:
        ax.axis('off')
    ax.set_title(title, fontsize=FSFS)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FSFS)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FSFS)
    fig.text(0.5, 0.12, subtext, horizontalalignment='center')
    fig.text(0.5, 0.05, subsubtext, horizontalalignment='center')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(scsc, cax=cbar_ax)
    fig.text(0.5, 0.98, title2, horizontalalignment='center', verticalalignment='top')

    if outpath is not None and outpath != 'none':
        print 'outputting matrix image to ', outpath
        plt.savefig(outpath + '.png')
    if show:
        plt.show()
    if close:
        plt.close()


def plot_real_matrix(M, name='', outpath=None, fig='auto', climv=None, cmap="coolwarm", show=False, close=True,
                     fontsize=None):
    """Plot matrix as colored subplot, with red positive and blue negative.

    Parameters
    ----------
    M : complex array
        matrix to plot
    name : string
        name to save plot WITHOUT extension (png)
    outpath : string (default='none' -> no saving)
        Directory and name of file as which to save plot. If outpath is None or 'none', does not save plot.
    show : bool (default == False)
        Whether to show the plot (and force user to close it to continue)
    clear : bool (default == True)
        Whether to clear the plot after saving or showing
    Returns
    ----------

    """
    if climv is None:
        climv = np.max(np.abs(M.ravel()))
        climvs = (-climv, climv)
    else:
        climvs = climv

    if fig == 'auto':
        fig = plt.gcf()
        plt.clf()

    a = fig.add_subplot(1, 1, 1)
    img = a.imshow(M, cmap="coolwarm", interpolation='none')
    a.set_title(name, fontsize=fontsize)
    cbar = plt.colorbar(img, orientation='horizontal')
    cbar.set_clim(climvs)

    if outpath is not None and outpath != 'none':
        print 'outputting matrix image to ', outpath
        plt.savefig(outpath + '.png')
    if show:
        plt.show()
    if close:
        plt.clf()


def plot_complex_matrix(M, name='', outpath=None, fig='auto', climvs=[], show=False, close=True, fontsize=None):
    """Plot real and imaginary parts of matrix as two subplots

    Parameters
    ----------
    M : complex array
        matrix to plot
    name : string
        name to save plot WITHOUT extension (png)
    outpath : string (default='none' -> no saving)
        Directory and name of file as which to save plot. If outpath is None or 'none', does not save plot.
    fig : matplotlib figure instance
        The figure to use for the plots
    clims : list of two lists
        Real and imaginary plot colorlimits, as [[real_lower, real_upper], [imag_lower, imag_upper]]
    show : bool (default == False)
        Whether to show the plot (and force user to close it to continue)
    close : bool (default == True)
        Whether to clear the plot after saving or showing
    fontsize : int
        The font size for the title, if name is not empty

    Returns
    ----------

    """
    # unpack or set colorlimit values
    if not climvs:
        climv_real_lower = -np.max(np.abs(np.real(M).ravel()))
        climv_real_upper = np.max(np.abs(np.real(M).ravel()))
        climv_imag_lower = -np.max(np.abs(np.imag(M).ravel()))
        climv_imag_upper = np.max(np.abs(np.imag(M).ravel()))
    else:
        climv_real_lower = climvs[0][0]
        climv_real_upper = climvs[0][1]
        climv_imag_lower = climvs[1][0]
        climv_imag_upper = climvs[1][1]

    if fig == 'auto':
        fig = plt.gcf()
        plt.clf()

    a = fig.add_subplot(1, 2, 1)
    img = a.imshow(np.imag(M), cmap="coolwarm", interpolation='none')
    a.set_title('Imaginary part ' + name, fontsize=fontsize)
    cbar = plt.colorbar(img, orientation='horizontal')
    cbar.set_clim(climv_imag_lower, climv_imag_upper)

    a = fig.add_subplot(1, 2, 2)
    img2 = a.imshow(np.real(M), cmap="coolwarm", interpolation='none')
    a.set_title('Real part ' + name, fontsize=fontsize)
    cbar = plt.colorbar(img2, orientation='horizontal')
    cbar.set_clim(climv_real_lower, climv_real_upper)

    if outpath is not None and outpath != 'none':
        print 'outputting complex matrix image to ', outpath
        plt.savefig(outpath + '.png')
    if show:
        plt.show()
    if close:
        plt.clf()


def absolute_sizer(ax=None):
    """Use the size of the matplotlib axis to create a function for sizing objects"""
    ppu = get_points_per_unit(ax)
    return lambda x: np.pi * (x*ppu)**2


def empty_scalar_mappable(vmin, vmax, cmap):
    """Create a scalar mappable for creating a colorbar that is not linked to specific data.

    Examples
    --------
    sm = empty_scalar_mappable(-1.0, 1.0, 'viridis')
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation=cbar_orientation, format=cbar_tickfmt)
    cbar.set_label(cax_label, labelpad=cbar_labelpad, rotation=0, fontsize=fontsize, va='center')

    Parameters
    ----------
    vmin : float
        the lower limit of the mappable
    vmax : float
        the upper limit of the mappable
    cmap : matplotlib colormap
        the colormap to use for the mappable
    """
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    return sm


def get_markerstyles(n):
    """Get a list of n marker style keys for matplotlib marker='' arguments, in a nicely defined order (so bizarre
    markers appear only for large n).
    If n < 28 (which is the number of 'nice' build-in markers, meaning ones I deem reasonably suitable for a plot,
    then all are distinct. Otherwise, some repeat."""
    all_markers = ['o', 'D', 's', '2', '*', 'h', '8', 'v', 'x', '+', 5, 'd', '>', 7, '.', '1', 'p', '3',
                   6, 0, 1, 2, 3, 4, '4', '<', 'H', '^']
    # Note: 0: 'tickleft', 1: 'tickright', 2: 'tickup', 3: 'tickdown', 4: 'caretleft', 'D': 'diamond', 6: 'caretup',
    #  7: 'caretdown', 's': 'square', '|': 'vline', '': 'nothing', 'None': 'nothing', 'x': 'x', 5: 'caretright',
    #  '_': 'hline', '^': 'triangle_up', ' ': 'nothing', 'd': 'thin_diamond', 'h': 'hexagon1', '+': 'plus', '*': 'star',
    #  ',': 'pixel', 'o': 'circle', '.': 'point', '1': 'tri_down', 'p': 'pentagon', '3': 'tri_left', '2': 'tri_up',
    #  '4': 'tri_right', 'H': 'hexagon2', 'v': 'triangle_down', '8': 'octagon', '<': 'triangle_left', None: 'nothing',
    #  '>': 'triangle_right'
    # all_markers = ['circle', 'diamond', 'square', 'tri_up', 'star',
    #                'hexagon1', 'octagon', 'triangle_down', 'x', 'plus',
    #                'caretright', 'thin_diamond', 'triangle_right', 'caretdown',
    #                'point', 'tri_down', 'pentagon', 'tri_left', 'caretup',
    #                'tickleft', 'tickright', 'tickup', 'tickdown', 'caretleft',
    #                'tri_right', 'triangle_left', 'hexagon2', 'triangle_up']
    # Note: markers can be found via
    # import matplotlib.pyplot as plt
    # import matplotlib
    # d = matplotlib.markers.MarkerStyle.markers
    # def find_symbol(syms):
    #     outsyms = []
    #     for sym in syms:
    #         for key in d:
    #             if d[key] == sym:
    #                 outsyms.append(key)
    #     return outsyms

    if n > len(all_markers):
        markerlist = all_markers
        markerlist.append(all_markers[0:n - len(all_markers)])
    else:
        markerlist = all_markers[0:n]

    return markerlist


def add_at(ax, t, loc=2):
    """Add attribute to a makeshift legend
    """
    from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
    fp = dict(size=8)
    _at = AnchoredText(t, loc=loc, prop=fp)
    ax.add_artist(_at)
    return _at


def annotate_connection_style(ax, x1y1, x2y2, connectionstyle="angle3,angleA=0,angleB=90",
                              xycoords='figure_fraction', textcoords='figure_fraction', color="0.5",
                              label=None):
    """Draw an annotation with a prescribed connection style on an axis.
    Example usage:
    demo_con_style(column[1], "angle3,angleA=0,angleB=90",
               label="angle3,\nangleA=0,\nangleB=90")
    See http://matplotlib.org/users/annotations_guide.html

    Parameters
    ----------
    ax : axis instance
        The axis on which to annotate
    x1y1 : tuple of floats
        coordinate pointed from
    x2y2 : tuple of floats
        coordinate pointed to
    connectionstyle : str
        Specifier for connection style, ex: "angle3,angleA=0,angleB=90"
    xycoords : str ('figure_fraction', 'axis_fraction', 'data')
        How to measure the location of the xy coordinates
    textcoords : str ('figure_fraction', 'axis_fraction', 'data')
        How to measure the location of the text coordinates
    color : color spec
        color of the arrow
    label : str or None
        label of annotation for a legend
    """
    if label is not None:
        add_at(ax, label, loc=2)

    x1, y1 = x1y1
    x2, y2 = x2y2

    ax.annotate("",
                xy=(x1, y1), xycoords=xycoords,
                xytext=(x2, y2), textcoords=textcoords,
                arrowprops=dict(arrowstyle="simple",  # linestyle="dashed",
                                color=color,
                                shrinkA=5,
                                shrinkB=5,
                                patchA=None,
                                patchB=None,
                                connectionstyle=connectionstyle,
                                ),
                )


def plot_pcolormesh(x, y, z, n, ax=None, cax=None, method='nearest', make_cbar=True, cmap='viridis',
                    vmin=None, vmax=None, title=None, xlabel=None, ylabel=None, ylabel_right=True,
                    ylabel_rot=90, cax_label=None, cbar_labelpad=3, cbar_orientation='vertical',
                    ticks=None, cbar_nticks=None, fontsize=12, title_axX=None, title_axY=None, alpha=1.0):
    """Interpolate x,y,z data onto an nxn meshgrid and plot as heatmap

    Parameters
    ----------
    x :
    y :
    z : float or int array

    n : int
        number of elements in each linear dimension in the meshgrid formed from x,y
    ax : axis instance or None
    cax : axis instance or None
        axis on which to plot the colorbar
    method : str
        interpolation specifier string
    make_cbar : bool
        Make a colorbar for the plot. If False, cax, cax_label, cbar_labelpad, cbar_orientation, and cbar_nticks are
        ignored
    cmap : colormap specifier
        The colormap to use for the pcolormesh
    vmin : float or None
        Color limit minimum value
    vmax : float or None
        Color limit maximum value
    title : str or None
        title for the plot
    titlepad : int or None
        Space above the plot to place the title
    alpha : float
        opacity
    """
    from lepm.data_handling import interpol_meshgrid
    X, Y, Z = interpol_meshgrid(x, y, z, n, method=method)
    if ax is None:
        ax = plt.gca()
    if cmap not in plt.colormaps():
        lecmaps.register_colormaps()
    pcm = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)

    if make_cbar:
        print 'making colorbar in plot_pcolormesh()...'
        cbar = plt.colorbar(pcm, cax=cax, orientation=cbar_orientation)
        if ticks is not None:
            cbar.set_ticks(ticks)
        elif cbar_nticks is not None:
            cbar.set_ticks(np.linspace(np.min(Z.ravel()), np.max(Z.ravel()), cbar_nticks))
        if cax_label is not None:
            if cbar_orientation == 'vertical':
                cbar.set_label(cax_label, labelpad=cbar_labelpad, rotation=0, fontsize=fontsize, va='center')
            else:
                cbar.set_label(cax_label, labelpad=cbar_labelpad, rotation=0, fontsize=fontsize)

    if title is not None:
        print '\n\n\nplotting.plotting: Making title\n\n\n'
        if title_axX is None and title_axY is None:
            ax.set_title(title, fontsize=fontsize)
        elif title_axX is None:
            print 'placing title at custom Y position...'
            ax.text(0.5, title_axY, title,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes)
        elif title_axY is None:
            print 'placing title at custom X position...'
            ax.text(title_axX, 1.0, title,
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    transform=ax.transAxes)
        else:
            print 'plotting.plotting: placing title at custom XY position...'
            ax.text(title_axX, title_axY, title,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, rotation=ylabel_rot)
        if ylabel_right:
            ax.yaxis.set_label_position("right")


def change_axes_geometry_stack(fig, ax, naxes):
    """Take a figure with stacked axes and add more stacked (in Y) axes to the figure, shuffling the others
    to make room for the new one(s).
    """
    for ii in range(len(ax)):
        geometry = (naxes,1,ii+1)
        if ax[ii].get_geometry() != geometry:
            ax[ii].change_geometry(*geometry)

    for ii in np.arange(len(ax),naxes):
        print 'adding axis ', ii
        fig.add_subplot(naxes, 1, ii+1)

    ax = fig.axes
    return fig, ax


def plot_axes_on_fig(ax, fig=None, geometry=(1, 1, 1)):
    """Attach ax to a new or different figure"""
    if fig is None:
        fig = plt.figure()
    if ax.get_geometry() != geometry :
        ax.change_geometry(*geometry)
    axes = fig.axes.append(ax)
    return fig, axes


def initialize_eigvect_DOS_header_plot(eigval, xy, sim_type='gyro', preset_cbar=False,
                                       page_orientation='portrait', ax_pos=[0.1, 0.10, 0.8, 0.60],
                                       cbar_pos=[0.79, 0.80, 0.012, 0.15],
                                       ax_cbar_pos=[0.9, 0.4, 0.012, 0.15], **kwargs):
    """initializes axes for the eigenvector and density of states plot.
    If preset_cbar is True, creates a colorbar, but does not return the handle.
    I think this was formerly called initialize_eigvect_DOS_plot

    Parameters
    ----------
    eigval : array of dimension 2nx1
        Eigenvalues of matrix for system
    xy : N x 2 float array
        Equilibrium positions of all the gyroscopes
    sim_type : str
        'gyro' 'mass(es?)' etc
    pin : int
        If positive, places text using this info... (look up how...)
    preset_cbar : bool
        If True, specifies colorbar position on plot using cbar_pos keywork argument to place the colorbar.
    orientation : str
        portrait or landscape, for figure aspect ratio
    cbar_pos : list of 4 floats
        xbottom ybottom, width and height of colorbar in figure_fraction coordinates
    kwargs : keyword arguments for colored_DOS_plot()
        For example, colorV = 1./ipr, DOSexcite=(2.25, 200.)

    Returns
    ----------
    fig : matplotlib figure
        figure with lattice and DOS drawn
    DOS_ax : axis instance
        axis for the DOS plot
    ax : axis instance
        axis for the time domain or other plots
    """
    if page_orientation == 'portrait':
        fig, ax, dos_ax, dos_cbar_ax, ax_cbar = \
            initialize_portrait_with_header(preset_cbar=preset_cbar, ax_pos=ax_pos,
                                            cbar_pos=cbar_pos, ax_cbar_pos=ax_cbar_pos)
    elif page_orientation == 'landscape':
        fig, ax, dos_ax, dos_cbar_ax, ax_cbar = \
            initialize_landscape_with_header(preset_cbar=preset_cbar, ax_pos=ax_pos,
                                             cbar_pos=cbar_pos, ax_cbar_pos=ax_cbar_pos)
    else:
        raise RuntimeError('Orientation parameter was not portrait or landscape')

    colored_DOS_plot(eigval, dos_ax, sim_type, alpha=1.0, cbar_ax=dos_cbar_ax, **kwargs)
    # DOS_plot(eigval, DOS_ax, sim_type)

    # Do this part also for absolute_sizer to work correctly on next plot call
    # Used to draw the lattice here:    lattice_plot(R, Ni, Nk, ax)
    # Eliminated this because we would have to pass the lattice attributes to this function
    ax.set_xlim(np.min(xy[:, 0]) - 1, np.max(xy[:, 0]) + 1)
    ax.set_ylim(np.min(xy[:, 1]) - 1, np.max(xy[:, 1]) + 1)

    return fig, dos_ax, ax


def initialize_portrait_with_header(preset_cbar=False, ax_pos=[0.1, 0.10, 0.8, 0.60],
                                    cbar_pos=[0.79, 0.80, 0.012, 0.15],
                                    ax_cbar_pos=[0.9, 0.4, 0.012, 0.15], ax_cbar=False):
    """

    Parameters
    ----------
    preset_cbar : bool
        If True, specifies colorbar position on plot using cbar_pos keywork argument to place the colorbar.
    orientation : str
        portrait or landscape, for figure aspect ratio
    cbar_pos : list of 4 floats
        xbottom ybottom, width and height of colorbar in figure_fraction coordinates
    kwargs : keyword arguments for colored_DOS_plot()
        For example, colorV = 1./ipr, DOSexcite=(2.25, 200.)

    Returns
    ----------
    fig : matplotlib figure
        figure with lattice and DOS drawn
    DOS_ax : axis instance
        axis for the DOS plot
    ax : axis instance
        axis for the time domain or other plots
    """
    fig = plt.figure(figsize=(1.5*5, 1.5*7))
    if preset_cbar:
        header_cbar = plt.axes(cbar_pos)
        header_ax = plt.axes([0.20, cbar_pos[1], 0.55, 0.15])
    else:
        header_ax = plt.axes([0.20, cbar_pos[1], 0.70, 0.15])
        header_cbar = None
    # axes constructor axes([left, bottom, width, height])
    ax = plt.axes(ax_pos)
    if ax_cbar:
        ax_cbar = plt.axes(ax_cbar_pos)
    else:
        ax_cbar = None

    return fig, ax, header_ax, header_cbar, ax_cbar


def initialize_landscape_with_header(preset_cbar=False, ax_pos=[0.1, 0.05, 0.8, 0.54],
                                     cbar_pos=[0.79, 0.80, 0.012, 0.15],
                                     ax_cbar_pos=[0.9, 0.4, 0.012, 0.], ax_cbar=False):
    """

    Parameters
    ----------
    preset_cbar : bool
        If True, specifies colorbar position on plot using cbar_pos keywork argument to place the colorbar.
    orientation : str
        portrait or landscape, for figure aspect ratio
    cbar_pos : list of 4 floats
        xbottom ybottom, width and height of colorbar in figure_fraction coordinates
    kwargs : keyword arguments for colored_DOS_plot()
        For example, colorV = 1./ipr, DOSexcite=(2.25, 200.)

    Returns
    ----------
    fig : matplotlib figure
        figure with lattice and DOS drawn
    DOS_ax : axis instance
        axis for the DOS plot
    ax : axis instance
        axis for the time domain or other plots
    """
    fig = plt.figure(figsize=(16. * 7.6 / 16., 9. * 7.6 / 16.))
    if preset_cbar:
        header_cbar = plt.axes(cbar_pos)
        header_ax = plt.axes([0.30, cbar_pos[1], 0.45, 0.18])
    else:
        header_ax = plt.axes([0.30, cbar_pos[1], 0.50, 0.18])
        header_cbar = None
    # axes constructor axes([left, bottom, width, height])
    ax = plt.axes(ax_pos)
    if ax_cbar:
        ax_cbar = plt.axes()
    else:
        ax_cbar = None

    return fig, ax, header_ax, header_cbar, ax_cbar


def initialize_lattice_DOS_header_plot(eigval, R, Ni, Nk, sim_type='gyro', preset_cbar=False,
                                       cbar_pos=[0.79, 0.78, 0.012, 0.15], **kwargs):
    """initializes axes for the eigenvalue/density of states plot.  Calls functions to draw lattice and DOS header.
    If preset_cbar is True, creates a colorbar, but does not return the handle for the colorbar.

    Parameters
    ----------
    eigval : array of dimension 2nx1
        Eigenvalues of matrix for system
    R : matrix of dimension nx3
        Equilibrium positions of all the gyroscopes
    Ni : matrix of dimension n x (max number of neighbors)
            Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes
    Nk : matrix of dimension n x (max number of neighbors)
            Correponds to Ni matrix.  1 corresponds to true connection while 0 signifies that there is not a connection
    pin : int
        If positive, places text using this info... (look up how...)
    preset_cbar : bool
        If True, specifies colorbar position on plot using cbar_pos keywork argument to place the colorbar.
    cbar_pos : list of 4 floats
        xbottom ybottom, width and height of colorbar in figure_fraction coordinates
    kwargs : keyword arguments for colored_DOS_plot()
        For example, colorV = 1./ipr, DOSexcite=(2.25, 200.)

    Returns
    ----------
    fig : matplotlib figure
        figure with lattice and DOS drawn
    DOS_ax : axis instance
        axis for the DOS plot
    ax : axis instance
        axis for the time domain or other plots
        """
    fig = plt.figure(figsize=(1.5*5, 1.5*5))

    # axes constructor axes([left, bottom, width, height])
    ax = plt.axes([0.01, 0.01, 0.98, 0.69])
    if preset_cbar:
        cbar_ax = plt.axes(cbar_pos)
        DOS_ax = plt.axes([0.20, 0.75, 0.55, 0.21])
    else:
        DOS_ax = plt.axes([0.20, 0.75, 0.7, 0.21])
        cbar_ax = None

    colored_DOS_plot(eigval, DOS_ax, sim_type, alpha=1.0, cbar_ax=cbar_ax, **kwargs)
    # DOS_plot(eigval, DOS_ax, sim_type)

    # Do this part also for absolute_sizer to function on next plot call
    lattice_plot(R, Ni, Nk, ax)

    return fig, DOS_ax, ax


def initialize_eigenvalue_DOS_plot(eigval, R, Ni, Nk, sim_type, pin):
    """PLEASE CONSIDER USING initialize_eigvect_DOS_header_plot INSTEAD!
    initializes axes for the eigenvalue/density of states plot.  calls functions to draw lattice and DOS

    Parameters
    ----------
    eigval : array of dimension 2nx1
        Eigenvalues of matrix for system

    R : matrix of dimension nx3
        Equilibrium positions of all the gyroscopes

    Ni : matrix of dimension n x (max number of neighbors)
            Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes

    Nk : matrix of dimension n x (max number of neighbors)
            Correponds to Ni matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection

    Returns
    ----------
    fig :
        figure with lattice and DOS drawn
    DOS_ax:
        axis for the DOS plot

    eig_ax
        axis for the eigenvalue plot
    """
    fig = plt.figure(figsize=(1.5*5, 1.5*7))

    DOS_ax = plt.axes([0.1, 0.75, 0.8, 0.20])  # axes constructor axes([left, bottom, width, height])
    eig_ax = plt.axes([0.1, 0.20, 0.8, 0.45])

    DOS_plot(eigval, DOS_ax, sim_type)
    lattice_plot(R, Ni, Nk, eig_ax)

    return fig, DOS_ax, eig_ax


def initialize_DOS_plot(eigval, sim_type, **kwargs):
    """initializes axes for the eigenvalue/density of states plot.  calls functions to draw lattice and DOS

    Parameters
    ----------
    eigval : array of dimension 2nx1
        Eigenvalues of matrix for system
    sim_type : str
        'gyro', 'mass', etc
    **kwargs: DOS_plot() keyword arguments


    Returns
    ----------
    fig :
        figure with lattice and DOS drawn
    DOS_ax:
        axis for the DOS plot

    eig_ax
        axis for the eigenvalue plot
    """
    fig = plt.figure(figsize=(10 * 0.6, 5 * 0.6))
    DOS_ax = plt.axes([0.15, 0.25, 0.8, 0.65])  # axes constructor axes([left, bottom, width, height])
    DOS_plot(eigval, DOS_ax, sim_type, **kwargs)
    # alpha=alpha, colorV=colorV, colormap=colormap,linewidth=linewidth)

    return fig, DOS_ax


def initialize_colored_DOS_plot(eigval, sim_type, axis_pos=(0.15, 0.25, 0.8, 0.65), **kwargs):
    """initializes axes for the eigenvalue/density of states plot.  calls functions to draw lattice and DOS

    Parameters
    ----------
    eigval : array of dimension 2nx1
        Eigenvalues of matrix for system
    sim_type : str
        'gyro', 'haldane', etc
    axis_pos : tuple
        the left, lower, right, and upper coordinates of the axis in units of figure size
    **kwargs: colored_DOS_plot() keyword arguments

    Returns
    ----------
    fig :
        figure with lattice and DOS drawn
    dos_ax:
        axis for the DOS plot
    eig_ax
        axis for the eigenvalue plot
        """
    fig = plt.figure(figsize=(10*0.6, 5*0.6))

    # note: syntax for axes constructor is axes([left, bottom, width, height])
    dos_ax = plt.axes(axis_pos)
    dos_ax, cbar_ax, cbar, n, bins = colored_DOS_plot(eigval, dos_ax, sim_type, **kwargs)

    return fig, dos_ax, cbar_ax, cbar


def initialize_colored_DOS_plot_twinax(eigval, sim_type, axis_pos=(0.15, 0.25, 0.8, 0.65), **kwargs):
    """initializes axes for the eigenvalue/density of states plot.  calls functions to draw lattice and DOS

    Parameters
    ----------
    eigval : array of dimension 2nx1
        Eigenvalues of matrix for system
    sim_type : str
        'gyro', 'haldane', etc
    axis_pos : tuple
        the left, lower, right, and upper coordinates of the axis in units of figure size
    **kwargs: colored_DOS_plot() keyword arguments

    Returns
    ----------
    fig :
        figure with lattice and DOS drawn
    dos_ax:
        axis for the DOS plot
    eig_ax
        axis for the eigenvalue plot
        """
    fig = plt.figure(figsize=(10*0.6, 5*0.6))

    # note: syntax for axes constructor is axes([left, bottom, width, height])
    dos_ax = plt.axes(axis_pos)
    dos_ax, cbar_ax, cbar, n, bins = colored_DOS_plot(eigval, dos_ax, sim_type, **kwargs)
    pos1 = dos_ax.get_position()  # get the original position
    pos2 = [pos1.x0, pos1.y0, pos1.width - 0.1, pos1.height]
    dos_ax.set_position(pos2)
    twin_ax = dos_ax.twinx()
    return fig, dos_ax, twin_ax, cbar_ax, cbar


def lattice_plot(R, Ni, Nk, ax, linecolor='k'):
    """draws lines for the gyro lattice (white lines connecting points)

    Parameters
    ----------
    R : matrix of dimension nx3
        Equilibrium positions of all the gyroscopes
    Ni : matrix of dimension n x (max number of neighbors)
        Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes
    Nk : matrix of dimension n x (max number of neighbors)
        Correponds to Ni matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection
    ax: python axis
        axis on which to draw lattice
    """
    plt.sca(ax)
    ax.set_aspect('equal')
    ax.set_axis_bgcolor('#d9d9d9')

    Rx = R[:, 0]
    Ry = R[:, 1]

    plt.xlim(Rx.min()-1, Rx.max()+1)
    plt.ylim(Ry.min()-1, Ry.max()+1)
    ax.set_autoscale_on(False)

    ppu = get_points_per_unit()
    s = absolute_sizer()

    CR = 0.4
    LW = 0.1

    NP = len(R)
    try:
        NN = np.shape(Ni)[1]
    except IndexError:
        '''There is only one particle'''
        NN = 0

    for i in range(NP):
        if NN > 0:
            for j, k in zip(Ni[i], Nk[i]):
                if i < j and k > 0:
                    ax.plot(R[(i, j), 0], R[(i, j), 1], linecolor, linewidth=LW*ppu, zorder=0)


def initialize_histogram(xx, alpha=1.0, colorV=None,  facecolor='#80D080', nbins=75,
                         fontsize=8, linewidth=1, xlabel=None, ylabel=None):
    """draws a histogram (such as DOS plot), where each bin can be colored according to colorV.

    Parameters
    ----------
    eigval : int or float array of dimension nx1
        values to histogram
    DOS_ax: matplotlib axis instance
        axis on which to draw the histogram
    alpha: float
        Opacity value for the bars on the plot
    colorV: len(eigval) x 1 float array
        values in (0,1) to translate into colors from colormap. Values outside the range (0,1) will be ducked as 0 or 1.
    colormap: str or matplotlib.colors.Colormap instance
        The colormap to use to determine the bin colors in the histogram
    facecolor: basestring or color specification
        hexadecimal or other specification of the color of the bars on the plot.
        Only used if colorV=None, otherwise colors will be based on colormap.

    """
    fig = plt.figure(figsize=(10*0.6, 5*0.6))
    hist_ax = plt.axes([0.15, 0.25, 0.8, 0.65])  # axes constructor axes([left, bottom, width, height])
    draw_histogram(xx, hist_ax, alpha=alpha, colorV=colorV, facecolor=facecolor, nbins=nbins,
                   fontsize=fontsize, linewidth=linewidth, xlabel=xlabel, ylabel=ylabel)
    return fig, hist_ax


def draw_histogram(xx, hist_ax, alpha=1.0, colorV=None, facecolor='#80D080', edgecolor=None, nbins=75,
                   fontsize=8, linewidth=1, xlabel=None, ylabel=None, label=None):
    """draws histogram (such as DOS plot), where each bin can be colored according to colorV.

    Parameters
    ----------
    xx : int or float array of dimension nx1
        values to histogram
    hist_ax: matplotlib axis instance
        axis on which to draw the histogram
    alpha: float
        Opacity value for the bars on the plot
    colorV: len(eigval) x 1 float array
        values in (0,1) to translate into colors from colormap. Values outside the range (0,1) will be ducked as 0 or 1.
    colormap: str or matplotlib.colors.Colormap instance
        The colormap to use to determine the bin colors in the histogram
    facecolor: basestring or color specification
        hexadecimal or other specification of the color of the bars on the plot.
        Only used if colorV=None, otherwise colors will be based on colormap.
    nbins : int
        The number of bins to make in the histogram
    fontsize : int
        The fontsize for the labels
    linewidth : float or int
        The width of the line outlining the histogram bins
    xlabel : str
        The label for the x axis
    ylabel : str
        The label for the y axis
    label : str
        The label for the legend of the histogram bins added here
    """
    plt.sca(hist_ax)
    if colorV is None:
        n, bins, patches = hist_ax.hist(xx, nbins, histtype='stepfilled', alpha=alpha, linewidth=linewidth, label=label)
        plt.setp(patches, 'facecolor', facecolor)
        if edgecolor is not None:
            plt.setp(patches, 'edgecolor', edgecolor)
    else:
        n, bins, patches = hist_ax.hist(xx, nbins, alpha=alpha, linewidth=linewidth, label=label)

    if xlabel is not None:
        hist_ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        hist_ax.set_ylabel(ylabel, fontsize=fontsize)
    return hist_ax


def DOS_plot(eigval, DOS_ax, sim_type, pin=-5000, alpha=1.0, colorV=None, facecolor='#80D080', nbins=75, fontsize=12,
             linewidth=1, xlabel=r'Oscillation frequency, $\omega$', ylabel=r'$D(\omega)$'):
    """draws DOS plot, where each bin can be colored according to colorV.

    Parameters
    ----------
    eigval : array of dimension 2nx1
        Eigenvalues of matrix for system
    DOS_ax: python axis
        axis on which to draw the DOS
    sim_type : str
        'gyro' 'mass', etc
    pin: float
        If pin != -5000, annotates the plot with the Omega_g value supplied
    alpha: float
        Opacity value for the bars on the plot
    colorV: len(eigval) x 1 float array
        values in (0,1) to translate into colors from colormap. Values outside the range (0,1) will be ducked as 0 or 1.
    colormap: str or matplotlib.colors.Colormap instance
        The colormap to use to determine the bin colors in the histogram
    facecolor: basestring or color specification
        hexadecimal or other specification of the color of the bars on the plot.
        Only used if colorV=None, otherwise colors will be based on colormap.

    """
    plt.sca(DOS_ax)
    # num_eigvals = len(eigval)

    if sim_type == 'gyro' or sim_type == 'magnetic_gyro' or sim_type == 'gHST_massive':
        hist_vals = np.abs(np.imag(eigval))
    else:
        hist_vals = np.real(eigval)

    if colorV is None:
        n, bins, patches = plt.hist(hist_vals, nbins, histtype='stepfilled', alpha=alpha, linewidth=linewidth)
    else:
        n, bins, patches = plt.hist(hist_vals, nbins, alpha=alpha, linewidth=linewidth)

    plt.setp(patches, 'facecolor', facecolor)

    # font = { 'family' : 'normal',
    #     'weight' : 'normal',
    #     'size'   : FSFS}
    # matplotlib.rc('font', **font)

    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    if sim_type == 'mass':
        nlim = np.partition(n, 4)[-3] + max(2, int(len(n)*0.05))
        plt.ylim(0, nlim)
    else:
        plt.ylim(0, max(n)+2)
    plt.xlim(min(bins), max(bins))
    # plt.yticks([0,20, 40, 60])
    if pin != -5000:
        # DON'T Convert to Hz (from omega value)
        # pin = pin / (2. * pi )
        DOS_ax.annotate('$\Omega_{g}$ = %0.2f rad/s' % pin, xy=(6., 9))


def excitation_DOS_plot(eigval, DOSexcite, DOS_ax=None, **kwargs):
    """Plot Gaussian excitation in kspace on top of DOS.

    Parameters
    ----------
    eigval: #modes x 1 complex or float array
        the eigenvalues of the system
    DOSexcite: tuple of floats or None
        (excitation frequency, stdev time), or else None if DOS plot is not desired.
        stdev time is conventionally excite_sigmatime
    DOS_ax: matplotlib axis instance or None
        the axis on which to plot the DOS and the Gaussian excitation spectrum
    **kwargs: DOS_plot() keyword arguments

    Returns
    -------
    DOS_ax
    """
    if DOS_ax is None:
        DOS_ax = plt.gca()

    DOS_plot(eigval, DOS_ax, **kwargs)

    # DOSexcite = (frequency, sigma_time)
    # amp(x) = exp[- acoeff * time**2]
    # amp(k) = sqrt(pi/acoeff) * exp[- pi**2 * k**2 / acoeff]
    # So 1/(2 * sigma_freq**2) = pi**2 /acoeff
    # So sqrt(acoeff/(2 * pi**2)) = sigma_freq

    sigmak = 1./DOSexcite[1]
    xlims = DOS_ax.get_xlim()
    ktmp = np.linspace(xlims[0], xlims[1], 300)
    gaussk = 0.8 * DOS_ax.get_ylim()[1] * np.exp(-(ktmp-DOSexcite[0])**2 / (2. * sigmak))
    DOS_ax.plot(ktmp, gaussk, 'r-')
    plt.sca(DOS_ax)

    return DOS_ax


def colored_DOS_plot(eigval, DOS_ax, sim_type, alpha=1.0, colorV=None, colormap='viridis', norm=None,
                     facecolor='#80D080', nbins=75, fontsize=12, cbar_ax=None, vmin=None, vmax=None, linewidth=1,
                     make_cbar=True, climbars=True, xlabel='Oscillation frequency, $\omega/\Omega_g$',
                     xlabel_pad=16, ylabel=r'$D(\omega)$', ylabel_pad=10, ylabel_ha='center', ylabel_va='center',
                     cax_label='', cbar_labelpad=10, ticks=None, cbar_nticks=None, cbar_tickfmt=None,
                     cbar_ticklabels=None,
                     orientation='vertical', cbar_orientation='vertical',
                     invert_xaxis=False, yaxis_tickright=False, yaxis_ticks=None, ylabel_right=False, ylabel_rot=90,
                     DOSexcite=None, DOSexcite_color='r', histrange=None, xlim=None, ylim=None):
    """draws DOS plot, where each bin can be colored according to colorV.

    Parameters
    ----------
    eigval : array of dimension 2nx1
        Eigenvalues of matrix for system
    DOS_ax: python axis
        axis on which to draw the DOS
    pin: float
        If pin != -5000, annotates the plot with the Omega_g value supplied
    alpha: float
        Opacity value for the bars on the plot
    colorV: len(eigval) x 1 float array
        values in (0,1) to translate into colors from colormap. Values outside the range (0,1) will be ducked as 0 or 1.
    colormap: str or matplotlib.colors.Colormap instance
        The colormap to use to determine the bin colors in the histogram
    norm:
    facecolor: basestring or color specification
        hexadecimal or other specification of the color of the bars on the plot.
        Only used if colorV=None, otherwise colors will be based on colormap.
    nbins: int
        Number of bins for histogram
    fontsize: int
        font size for labels and title
    cbar_ax: matplotlib axis instance or None
        axis to use for the colorbar
    vmin: float or None
        The lower bound for the colorbar/colormap. If None, uses auto limits
    vmax: float or None
        The upper bound for the colorbar/colormap. If None, uses auto limits
    linewidth: float
        width of the line separating bars, or, if colorV is None, the line above the bars.
    make_cbar: bool
        Whether to show the colorbar at all.
    climbars: bool
        If vmin is None and vmax is None, determines whether the max of colorV or max histogram bar value (based on
        colorV) is used for the maximum color value
    xlabel : str
    ylabel : str
    ylabel_pad : float or int
        Space between ylabel and plot
    cax_label: str
        Label for the colorbar, if make_cbar is True
    cbar_labelpad : int or float
        Space between colorbar and colorbarlabel
    ticks : list of floats or ints or None
        tick positions for the colorbar
    cbar_nticks : int or None
        if not None, will set cbar_nticks evenly spaced for colorbar
    cbar_tickfmt : None or format specification
        format for the tick labels on the colorbar, for example
    orientation : str specifier (default='vertical')
    cbar_orientation : str specifier (default='vertical')
    invert_xaxis : bool
        Whether or not to invert the x axis
    yaxis_tickright : bool
    yaxis_ticks : list of floats or ints, or None
        The values to be specified on the y axis
    ylabel_right : bool
        Put the ylabel on the RHS of the plot
    ylabel_rot : float
        Angle by which to rotate the ylabel
    DOSexcite: tuple of floats or None
        (excitation frequency, stdev time), or else None if plotting an excitation on the DOS plot is not desired.
        stdev time is conventionally named excite_sigmatime elsewhere
    DOSexcite_color : color spec
        Color for the curve in the DOS showing the excitation

    Returns
    ------
    DOS_ax, cbar_ax, cbar, n, bins
    """
    if nbins is None:
        nbins = 75

    if colormap not in plt.colormaps():
        lecmaps.register_colormaps()

    print '\nCOLORMAP = ', colormap

    plt.sca(DOS_ax)
    # num_eigvals = len(eigval)

    if sim_type == 'gyro' or sim_type == 'magnetic_gyro' or sim_type == 'gHST_massive':
        # hist_vals = array([imag(eigval[i]) for i in range(len(eigval)) if abs(real(eigval[i])) < 1])
        hist_vals = np.abs(np.imag(eigval))
    else:
        hist_vals = np.real(eigval)

    if colorV is None:
        n, bins, patches = DOS_ax.hist(hist_vals, nbins, histtype='stepfilled', alpha=alpha, linewidth=linewidth,
                                       orientation=orientation, range=histrange)
    else:
        n, bins, patches = DOS_ax.hist(hist_vals, nbins, alpha=alpha, linewidth=linewidth, orientation=orientation,
                                       range=histrange)

    if colorV is not None:
        print 'plotting.plotting: np.shape(colorV) = ', np.shape(colorV)
        if len(colorV) == nbins:
            colors = colorV
        elif len(colorV) == len(eigval):
            # bin colorV to match nbins
            # digits gives the indices of the bin to which each value in histogram belongs
            digits = np.digitize(hist_vals, bins)
            # digits = bins.searchsorted(hist_vals, 'right')

            # For each bin, average colorV[ digits == bin# ]
            colors = np.zeros(nbins, dtype=float)
            for kk in range(nbins):
                if kk == 0:
                    if (digits < kk + 2).any():
                        colors[kk] = np.mean(colorV[digits < kk + 2])
                elif kk == nbins - 1:
                    if (digits > kk).any():
                        colors[kk] = np.mean(colorV[digits > kk])
                else:
                    if (digits == kk + 1).any():
                        colors[kk] = np.mean(colorV[digits == kk + 1])
        else:
            raise RuntimeError("Supplied colorV must have same length as eigval or as nbins (#bins).")

        # If there is a normalization specified, consider that here. UNFINISHED!
        if norm is None:
            if vmin is None:
                if climbars:
                    vmin = np.min(colors)
                else:
                    vmin = np.min(colorV)
            if vmax is None:
                if climbars:
                    vmax = np.max(colors)
                    print 'vmax --> climbars:', vmax
                else:
                    vmax = np.max(colorV)
                    print 'vmax --> climbars:', vmin
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            sm = matplotlib.cm.ScalarMappable(cmap=colormap, norm=norm)
        # else:
        #     norm = mpl.colors.LogNorm(vmin=0.0, vmax=1.0)
        #     sm = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
        #     sm.set_clim([0.0,1.0])
        # colormap = matplotlib.cm.get_cmap(name=colormap)

        for colorii, patchii in zip(colors, patches):
            color = sm.to_rgba(colorii)
            patchii.set_facecolor(color)

        sm._A = []
        if make_cbar:
            print 'making cbar...'
            cbar = plt.colorbar(sm, cax=cbar_ax, orientation=cbar_orientation, format=cbar_tickfmt)
            cbar.set_label(cax_label, labelpad=cbar_labelpad, rotation=0, fontsize=fontsize, va='center')
            if ticks is not None:
                cbar.set_ticks(ticks)
            elif cbar_nticks is not None:
                cbar.set_ticks(np.linspace(vmin, vmax, cbar_nticks))
            if cbar_ticklabels is not None:
                cbar.set_ticklabels(cbar_ticklabels)
        else:
            cbar = None
    else:
        plt.setp(patches, 'facecolor', facecolor)
        cbar = None

    # font = { 'family' : 'normal',
    #     'weight' : 'normal',
    #     'size'   : FSFS}
    # matplotlib.rc('font', **font)

    if orientation == 'vertical':
        if xlabel is not None:
            DOS_ax.set_xlabel(xlabel, fontsize=fontsize, ha=ylabel_ha, va=ylabel_va, labelpad=xlabel_pad)
        if ylabel is not None:
            DOS_ax.set_ylabel(ylabel, fontsize=fontsize, rotation=ylabel_rot, ha=ylabel_ha, va=ylabel_va,
                              labelpad=ylabel_pad)
        # if ylabel_pad is not None:
        #     DOS_ax.yaxis.labelpad = ylabel_pad
        # if xlabel_pad is not None:
        #     DOS_ax.xaxis.labelpad = xlabel_pad
        if sim_type == 'mass':
            nlim = np.partition(n, 4)[-3] + max(2, int(len(n)*0.05))
            DOS_ax.set_ylim(0, nlim)
        else:
            DOS_ax.set_ylim(0, max(n)+2)

        DOS_ax.set_xlim(min(bins), max(bins) + min(.01, 0.01*(np.max(bins) - np.min(bins))))
    else:
        if ylabel is not None:
            DOS_ax.set_xlabel(ylabel, fontsize=fontsize, ha=ylabel_ha, va=ylabel_va, labelpad=xlabel_pad)
        if xlabel is not None:
            DOS_ax.set_ylabel(xlabel, fontsize=fontsize, rotation=ylabel_rot, ha=ylabel_ha, va=ylabel_va,
                              labelpad=ylabel_pad)
        # if ylabel_pad is not None:
        #     DOS_ax.yaxis.labelpad = ylabel_pad
        # if xlabel_pad is not None:
        #     DOS_ax.xaxis.labelpad = xlabel_pad
        if sim_type == 'mass':
            nlim = np.partition(n, 4)[-3] + max(2, int(len(n) * 0.05))
            DOS_ax.set_xlim(0, nlim)
        else:
            DOS_ax.set_xlim(0, max(n) + 2)

        DOS_ax.set_ylim(min(bins), max(bins))

    if invert_xaxis:
        DOS_ax.invert_xaxis()

    if yaxis_tickright:
        DOS_ax.yaxis.tick_right()

    if yaxis_ticks is not None:
        DOS_ax.set_yticks(yaxis_ticks)

    if ylabel_right:
        print '\n\n\n\nplotting.plotting: setting ylabel position to be right\n\n\n\n'
        DOS_ax.yaxis.set_label_position("right")

    # plt.yticks([0,20, 40, 60])
    # if pin != -5000:
    #     # DON'T Convert to Hz (from omega value)
    #     # pin = pin/(2.*pi)
    #     DOS_ax.annotate('$\Omega_{g}$ = %0.2f rad/s' % pin, xy=(6., 9))

    # Add excitation Gaussian curve
    if DOSexcite is not None:
        # DOSexcite = (frequency, sigma_time)
        # amp(x) = exp[- acoeff * time**2]
        # amp(k) = sqrt(pi/acoeff) * exp[- pi**2 * k**2 / acoeff]
        # So 1/(2 * sigma_freq**2) = pi**2 /acoeff
        # So sqrt(acoeff/(2 * pi**2)) = sigma_freq

        sigmak = 1./DOSexcite[1]
        xlims = DOS_ax.get_xlim()
        ktmp = np.linspace(xlims[0], xlims[1], 300)
        gaussk = 0.8 * DOS_ax.get_ylim()[1] * np.exp(-(ktmp-DOSexcite[0])**2 / (2. * sigmak))
        DOS_ax.plot(ktmp, gaussk, '-', color=DOSexcite_color)

    if xlim is not None:
        DOS_ax.set_xlim(xlim)
    if ylim is not None:
        DOS_ax.set_ylim(ylim)

    return DOS_ax, cbar_ax, cbar, n, bins


def shaded_DOS_plot(eigval, DOS_ax, sim_type, alpha=None, facecolor='#80D080', nbins=75,
                    fontsize=12, cbar_ax=None, vmin=None, vmax=None, linewidth=1, cax_label='',
                    make_cbar=True, climbars=True, xlabel=r'Oscillation frequency $\omega$', ylabel='number of states',
                    ticks=None):
    """draws DOS plot, where each bin can be colored according to colorV.

    Parameters
    ----------
    eigval : array of dimension 2nx1
        Eigenvalues of matrix for system
    DOS_ax: python axis
        axis on which to draw the DOS
    pin: float
        If pin != -5000, annotates the plot with the Omega_g value supplied
    alpha: nbins x 1 or len(eigval) x 1 float array
        Opacity value for each bar or each eigval in the plot
    colorV: len(eigval) x 1 float array
        values in (0,1) to translate into colors from colormap. Values outside the range (0,1) will be ducked as 0 or 1.
    colormap: str or matplotlib.colors.Colormap instance
        The colormap to use to determine the bin colors in the histogram
    facecolor: basestring or color specification
        hexadecimal or other specification of the color of the bars on the plot.
        Only used if colorV=None, otherwise colors will be based on colormap.
    nbins: int
        Number of bins for histogram
    fontsize: int
        font size for labels and title
    cbar_ax: matplotlib axis instance or None
        axis to use for the colorbar
    vmin: float or None
        The lower bound for the colorbar/colormap
    vmax: float or None
        The upper bound for the colorbar/colormap
    linewidth: float
        width of the line separating bars, or, if colorV is None, the line above the bars.
    cax_label: str
        Label for the colorbar, if make_cbar is True
    make_cbar: bool
        Whether to show the colorbar at all.
    climbars: bool
        If vmin is None and vmax is None, determines whether the max of colorV or max histogram bar value (based on
        colorV) is used for the maximum color value

    Returns
    ------
    DOS_ax, cbar_ax, cbar
    """
    # plt.sca(DOS_ax)
    # num_eigvals = len(eigval)

    if sim_type == 'gyro' or sim_type == 'magnetic_gyro' or sim_type == 'gHST_massive':
        # hist_vals = array([imag(eigval[i]) for i in range(len(eigval)) if abs(real(eigval[i])) < 1])
        hist_vals = np.array([np.abs(np.imag(eigval[i])) for i in range(len(eigval))])
    else:
        hist_vals = np.array([np.real(eigval[i]) for i in range(len(eigval))])

    if alpha is None:
        n, bins, patches = DOS_ax.hist(hist_vals, nbins, histtype='stepfilled', alpha=alpha, linewidth=linewidth)
    else:
        n, bins, patches = DOS_ax.hist(hist_vals, nbins, linewidth=linewidth)

    if alpha is not None:
        if len(alpha) == nbins:
            alphas = alpha
        elif len(alpha) == len(eigval):
            # bin colorV to match nbins
            # digits gives the indices of the bin to which each value in histogram belongs
            digits = np.digitize(hist_vals, bins)
            # digits = bins.searchsorted(hist_vals, 'right')

            # For each bin, average colorV[ digits == bin# ]
            alphas = np.zeros(nbins, dtype=float)
            for kk in range(nbins):
                if kk == 0:
                    if (digits < kk + 2).any():
                        alphas[kk] = np.mean(alpha[digits < kk + 2])
                elif kk == nbins - 1:
                    if (digits > kk).any():
                        alphas[kk] = np.mean(alpha[digits > kk ])
                else:
                    if (digits == kk + 1).any():
                        alphas[kk] = np.mean(alpha[digits == kk + 1])
        else:
            raise RuntimeError("Supplied alpha (array of opacities) must have same length as eigval or as nbins (#bins).")

        # Unpack the mins and maxes for the alpha values, either from max/min of bins or max/min of supplied alpha vect
        if vmin is None:
            if climbars:
                vmin = np.min(alphas)
            else:
                vmin = np.min(alpha)
        if vmax is None:
            if climbars:
                vmax = np.max(alphas)
            else:
                vmax = np.max(alpha)
        # print 'vmax = ', vmax

        alphaV = (alphas - float(vmin))/(float(vmax) - float(vmin))
        alphaV = np.amin(np.dstack((alphaV, np.ones_like(alphaV)))[0], axis=1)
        # print 'alphaV = ', alphaV

        # if isinstance(facecolor,str):
        #    facecolor = colormaps.hex2rgbdecimal(facecolor)
        for alphaii, patchii in zip(alphaV, patches):
            # print 'facecolor = ', facecolor[0:3]
            # patchii.set_facecolor(facecolor[0:3])
            # print 'alphaii = ', alphaii
            patchii.set_alpha(alphaii)

        plt.setp(patches, 'facecolor', facecolor)

        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        colormap = lecmaps.colormap_from_hex()
        sm = matplotlib.cm.ScalarMappable(cmap=colormap, norm=norm)

        sm._A = []
        if make_cbar:
            print 'making cbar...'
            cbar = plt.colorbar(sm, cax=cbar_ax)
            cbar.set_label(cax_label, labelpad=10, rotation=0, fontsize=fontsize)
        else:
            cbar = None
    else:
        plt.setp(patches, 'facecolor', facecolor)
        cbar = None

    DOS_ax.set_xlabel(xlabel, fontsize=fontsize)
    DOS_ax.set_ylabel(ylabel, fontsize=fontsize)

    if sim_type == 'mass':
        nlim = np.partition(n, 4)[-3] + max(2, int(len(n)*0.05))
        DOS_ax.set_ylim(0, nlim)
    else:
        DOS_ax.set_ylim(0, max(n)+2)
    DOS_ax.set_xlim(min(bins), max(bins))
    # plt.yticks([0,20, 40, 60])
    if pin != -5000:
        # DON'T Convert to Hz (from omega value)
        # pin = pin/(2.*pi )
        DOS_ax.annotate('$\Omega_{g}$ = %0.2f rad/s' % pin, xy=(6., 9))

    return DOS_ax, cbar_ax, cbar, n, bins


def construct_eigvect_DOS_plot(xy, fig, DOS_ax, eig_ax, eigval, eigvect, en, sim_type, Ni, Nk, marker_num=0,
                               color_scheme='default', sub_lattice=-1, cmap_lines='BlueBlackRed', line_climv=None,
                               cmap_patches='isolum_rainbow', draw_strain=False, lw=0.8, bondval_matrix=None):
    """puts together lattice and DOS plots and draws normal mode ellipsoids on top

    Parameters
    ----------
    xy: array 2N x 3
        Equilibrium position of the gyroscopes
    fig :
        figure with lattice and DOS drawn
    DOS_ax:
        axis for the DOS plot
    eig_ax
        axis for the eigenvalue plot
    eigval : array of dimension 2nx1
        Eigenvalues of matrix for system
    eigvect : array of dimension 2nx2n
        Eigenvectors of matrix for system.
        Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
        x0, y0, x1, y1, ... xNP, yNP.
    en: int
        Number of the eigenvalue you are plotting

    Returns
    ----------
    fig :
        completed figure for normal mode

    [scat_fg, p, f_mark] :
        things to be cleared before next normal mode is drawn
        """
    ppu = get_points_per_unit()
    s = absolute_sizer()

    plt.sca(DOS_ax)

    ev = eigval[en]
    ev1 = ev

    # Show where current eigenvalue is in DOS plot
    (f_mark, ) = plt.plot([np.abs(ev), np.abs(ev)], plt.ylim(), '-r')

    NP = len(xy)

    im1 = np.imag(ev)
    re1 = np.real(ev)
    plt.sca(eig_ax)
    plt.title('Mode %d; $\Omega=( %0.6f + %0.6f i)$' % (en, re1, im1))

    # Preallocate ellipsoid plot vars
    shap = eigvect.shape
    angles_arr = np.zeros(NP)

    patch = []
    colors = np.zeros(NP+2)
    x_mag = np.zeros(NP)
    y_mag = np.zeros(NP)

    x0s = np.zeros(NP)
    y0s = np.zeros(NP)

    mag1 = eigvect[en]

    # Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
    # x0, y0, x1, y1, ... xNP, yNP.
    mag1x = np.array([mag1[2 * i] for i in range(NP)])
    mag1y = np.array([mag1[2 * i + 1] for i in range(NP)])

    # Pick a series of times to draw out the ellipsoid
    time_arr = np.arange(81) * 2 * np.pi/(np.abs(ev1) * 80)
    exp1 = np.exp(ev1 * time_arr)

    # Normalization for the ellipsoids
    lim_mag1 = np.max(np.array([np.sqrt(np.abs(exp1*mag1x[i])**2 + np.abs(exp1*mag1y[i])**2)
                                for i in range(len(mag1x))]).flatten())
    mag1x /= lim_mag1
    mag1y /= lim_mag1

    cw = []
    ccw = []
    lines_1 = []
    for i in range(NP):
        x_disps = 0.5 * (exp1*mag1x[i]).real
        y_disps = 0.5 * (exp1*mag1y[i]).real

        x_vals = xy[i, 0] + x_disps
        y_vals = xy[i, 1] + y_disps

        poly_points = np.array([x_vals, y_vals]).T
        polygon = Polygon(poly_points, True)

        # x0 is the marker_num^th element of x_disps
        x0 = x_disps[marker_num]
        y0 = y_disps[marker_num]

        # x0s is the position (global pos, not relative) of each gyro at time = marker_num(out of 81)
        x0s[i] = x_vals[marker_num]
        y0s[i] = y_vals[marker_num]

        # These are the black lines protruding from pivot point to current position
        lines_1.append([[xy[i, 0], x_vals[marker_num]], [xy[i, 1], y_vals[marker_num]]])

        mag = np.sqrt(x0**2+y0**2)
        if mag > 0:
            anglez = np.arccos(x0/mag)
        else:
            anglez = 0

        if y0 < 0:
            anglez = 2*np.pi-anglez

        # testangle = arctan2(y0, x0)
        # print '\n diff angles = ', anglez - testangle
        # print ' x0 - x_disps[0] =', x0-x_disps[marker_num]

        angles_arr[i] = anglez
        patch.append(polygon)

        # Do Fast Fourier Transform (FFT)
        # ff = abs(fft.fft(x_disps + 1j*y_disps))**2
        # ff_freq = fft.fftfreq(len(x_vals), 1)
        # mm_f = ff_freq[ff == max(ff)][0]

        if color_scheme == 'default':
            colors[i] = anglez
        else:
            if sub_lattice[i] == 0:
                colors[i] = 0
            else:
                colors[i] = np.pi
            ccw.append(i)

    colors[NP] = 0
    colors[NP+1] = 2 * np.pi

    plt.yticks([])
    plt.xticks([])
    # this is the part that puts a dot a t=0 point
    scat_fg = eig_ax.scatter(x0s[cw], y0s[cw], s=s(.02), c='DodgerBlue')
    scat_fg2 = eig_ax.scatter(x0s[ccw], y0s[ccw], s=s(.02), c='Red', zorder=3)

    NP = len(xy)
    try:
        NN = np.shape(Ni)[1]
    except IndexError:
        NN = 0

    Rnorm = np.array([x0s, y0s]).T
    # print 'Rnorm = ', np.shape(Rnorm)
    # print 'xy = ', np.shape(xy)
    # print 'Ni = ', Ni
    # print 'Nk = ', Nk

    # Bond Stretches
    if draw_strain:
        inc = 0
        stretches = np.zeros(4*len(xy))
        for i in range(len(xy)):
            if NN > 0:
                for j, k in zip(Ni[i], Nk[i]):
                    if i < j and abs(k) > 0:
                        n1 = float(np.linalg.norm(Rnorm[i] - Rnorm[j]))
                        n2 = np.linalg.norm(xy[i] - xy[j])
                        stretches[inc] = (n1 - n2)
                        inc += 1

        stretch = np.array(stretches[0:inc])
    else:
        # simply get length of BL by iterating over all bonds
        inc = 0
        for i in range(len(xy)):
            if NN > 0:
                for j, k in zip(Ni[i], Nk[i]):
                    if i < j and abs(k) > 0:
                        inc += 1

    # For particles with neighbors, get list of bonds to draw.
    # If bondval_matrix is not None, color by the elements of that matrix
    if bondval_matrix is not None or draw_strain:
        test = list(np.zeros([inc, 1]))
        bondvals = list(np.ones([inc, 1]))
        inc = 0
        xy = np.array([x0s, y0s]).T
        for i in range(len(xy)):
            if NN > 0:
                for j, k in zip(Ni[i], Nk[i]):
                    if i < j and abs(k) > 0:
                        test[inc] = [xy[(i, j), 0], xy[(i, j), 1]]
                        if bondval_matrix is not None:
                            bondvals[inc] = bondval_matrix[i, j]
                        inc += 1

            # lines connect sites (bonds), while lines_12 draw the black lines from the pinning to location sites
            lines = [zip(x, y) for x, y in test]

    # lines_12 draw the black lines from the pinning to location sites
    lines_12 = [zip(x, y) for x, y in lines_1]

    # Check that we have all the cmaps
    if cmap_lines not in plt.colormaps() or cmap_patches not in plt.colormaps():
        lecmaps.register_cmaps()

    # Add lines colored by strain here
    if bondval_matrix is not None:
        lines_st = LineCollection(lines, array=bondvals, cmap=cmap_lines, linewidth=0.8)
        if line_climv is None:
            maxk = np.max(np.abs(bondvals))
            mink = np.min(np.abs(bondvals))
            if (bondvals - bondvals[0] < 1e-8).all():
                lines_st.set_clim([mink - 1., maxk + 1.])
            else:
                lines_st.set_clim([mink, maxk])

        lines_st.set_zorder(2)
        eig_ax.add_collection(lines_st)
    else:
        if draw_strain:
            lines_st = LineCollection(lines, array=stretch, cmap=cmap_lines, linewidth=0.8)
            if line_climv is None:
                maxstretch = np.max(np.abs(stretch))
                if maxstretch < 1e-8:
                    line_climv = 1.0
                else:
                    line_climv = maxstretch

            lines_st.set_clim([-line_climv, line_climv])
            lines_st.set_zorder(2)
            eig_ax.add_collection(lines_st)

    lines_12_st = LineCollection(lines_12, linewidth=0.8)
    lines_12_st.set_color('k')

    p = PatchCollection(patch, cmap=cmap_patches, alpha=0.6)

    p.set_array(np.array(colors))
    p.set_clim([0, 2 * np.pi])
    p.set_zorder(1)

    eig_ax.add_collection(lines_12_st)
    eig_ax.add_collection(p)

    eig_ax.set_aspect('equal')

    # erased ev/(2*pi) here npm 2016
    cw_ccw = [cw, ccw, ev]
    # print cw_ccw[1]

    # If on a virtualenv, check it here
    # if not hasattr(sys, 'real_prefix'):
    #     plt.show()
    #     eig_ax.set_facecolor('#000000')
    #     print 'leplt: construct_eigvect_DOS_plot() exiting'

    return fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw


def plot_eigvect_excitation(xy, fig, dos_ax, eig_ax, eigval, eigvect, en, marker_num=0,
                            black_t0lines=False, mark_t0=True, title='auto', normalization=1., alpha=0.6,
                            lw=1, zorder=10):
    """Draws normal mode ellipsoids on axis eig_ax.
    If black_t0lines is true, draws the black line from pinning site to positions

    Parameters
    ----------
    xy: array 2N x 3
        Equilibrium position of the gyroscopes
    fig :
        figure with lattice and DOS drawn
    dos_ax: matplotlib axis instance or None
        axis for the DOS plot. If None, ignores this input
    eig_ax : matplotlib axis instance
        axis for the eigenvalue plot
    eigval : array of dimension 2nx1
        Eigenvalues of matrix for system
    eigvect : array of dimension 2nx2n
        Eigenvectors of matrix for system.
        Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
        x0, y0, x1, y1, ... xNP, yNP.
    en: int
        Number of the eigenvalue you are plotting
    marker_num : int in (0, 80)
        where in the phase (0 to 80) to call t=t0. This sets "now" for drawing where in the normal mode to draw
    black_t0lines : bool
        Draw black lines extending from the pinning site to the current site (where 'current' is determined by
        marker_num)

    Returns
    ----------
    fig : matplotlib figure instance
        completed figure for normal mode
    [scat_fg, pp, f_mark, lines12_st] :
        things to be cleared before next normal mode is drawn
        """
    # ppu = get_points_per_unit()
    s = absolute_sizer()

    ev = eigval[en]
    ev1 = ev

    # Show where current eigenvalue is in DOS plot
    if dos_ax is not None:
        (f_mark, ) = dos_ax.plot([abs(ev), abs(ev)], dos_ax.get_ylim(), '-r')

    NP = len(xy)

    im1 = np.imag(ev)
    plt.sca(eig_ax)

    if title == 'auto':
        eig_ax.set_title('$\omega = %0.6f$' % im1)
    elif title is not None and title not in ['', 'none']:
        eig_ax.set_title(title)

    # Preallocate ellipsoid plot vars
    angles_arr = np.zeros(NP)

    patch = []
    colors = np.zeros(NP)
    x0s = np.zeros(NP, dtype=float)
    y0s = np.zeros(NP, dtype=float)
    mag1 = eigvect[en]

    # Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
    # x0, y0, x1, y1, ... xNP, yNP.
    mag1x = np.array([mag1[2*i] for i in range(NP)])
    mag1y = np.array([mag1[2*i+1] for i in range(NP)])

    # Pick a series of times to draw out the ellipsoid
    time_arr = np.arange(81) * 2 * np.pi/(np.abs(ev1) * 80)
    exp1 = np.exp(ev1 * time_arr)

    # Normalization for the ellipsoids
    lim_mag1 = np.max(np.array([np.sqrt(np.abs(exp1*mag1x[i])**2 + np.abs(exp1*mag1y[i])**2)
                                for i in range(len(mag1x))]).flatten())
    mag1x /= lim_mag1
    mag1y /= lim_mag1
    mag1x *= normalization
    mag1y *= normalization

    if black_t0lines:
        lines_1 = []
    else:
        lines_12_st = []

    for i in range(NP):
        x_disps = 0.5 * (exp1*mag1x[i]).real
        y_disps = 0.5 * (exp1*mag1y[i]).real

        x_vals = xy[i, 0] + x_disps
        y_vals = xy[i, 1] + y_disps

        poly_points = np.array([x_vals, y_vals]).T
        polygon = Polygon(poly_points, True)

        # x0 is the marker_num^th element of x_disps
        x0 = x_disps[marker_num]
        y0 = y_disps[marker_num]

        x0s[i] = x_vals[marker_num]
        y0s[i] = y_vals[marker_num]

        if black_t0lines:
            # These are the black lines protruding from pivot point to current position
            lines_1.append([[xy[i, 0], x_vals[marker_num]], [xy[i, 1], y_vals[marker_num]]])

        mag = np.sqrt(x0**2 + y0**2)
        if mag > 0:
            anglez = np.arccos(x0/mag)
        else:
            anglez = 0

        if y0 < 0:
            anglez = 2 * np.pi - anglez

        angles_arr[i] = anglez
        patch.append(polygon)
        colors[i] = anglez

    # this is the part that puts a dot a t=0 point
    if mark_t0:
        scat_fg = eig_ax.scatter(x0s, y0s, s=s(.02), c='k')
    else:
        scat_fg = []

    pp = PatchCollection(patch, cmap='hsv', lw=lw, alpha=alpha, zorder=zorder)

    pp.set_array(np.array(colors))
    pp.set_clim([0, 2 * np.pi])
    pp.set_zorder(1)

    eig_ax.add_collection(pp)

    if black_t0lines:
        lines_12 = [zip(x, y) for x, y in lines_1]
        lines_12_st = LineCollection(lines_12, linewidth=0.8)
        lines_12_st.set_color('k')
        eig_ax.add_collection(lines_12_st)

    eig_ax.set_aspect('equal')

    return fig, [scat_fg, pp, f_mark, lines_12_st]


def clear_plot(figure, clear_array):
    """clears plot of items specified by clear_array

    Parameters
    ----------
    figure :
        figure with items to be cleared

    clear_array :
        items to clear from figure
    """
    for i in range(len(clear_array)):
        clear_array[i].remove()


def decomposition_with_inverse(X, eigvect):
    """Assumes eigvect will be in row form.  Assumes that X is a 2n x 1 array
    """
    vects = eigvect.T
    inv_vects = np.linalg.inv(vects)
    dim = np.shape(vects)  # vects should be 2n x 2n
    X = np.reshape(X, [dim[0], 1])
    icm = np.dot(inv_vects, X)
    return icm


def decomp_plot(fig, current_pos, eigvect, eigval, title_label, decomp_ax):
    """makes plot of mode decomposition for time step

    Parameters
    ----------
    fig :
        figure with lattice and DOS drawn
    current_pos: array 2nx3
        current positions of the gyros
    DOS_ax:
        axis for the DOS plot
    eig_ax
        axis for the eigenvalue plot
    eigvect : array of dimension 2nx2n
        Eigenvectors of matrix for system
    eigval : array of dimension 2nx1
        Eigenvalues of matrix for system
    title_label: string
        title for plot (It says 'Mode Spectrum for title_label_)
    decomp_ax : axis
        axis to draw decomposition on

    Returns
    ----------

    [l] :
        things to be cleared before next decomposition is drawn
        """

    plt.sca(decomp_ax)

    cp = np.array([current_pos[:, 0], current_pos[:, 1]]).T

    decomp = abs(decomposition_with_inverse(cp, eigvect))**2  # ones(len(eigval))
    si = np.argsort(eigval)
    decomp = decomp[si]
    eigval = eigval[si]  # /(2*pi) #erased 2 pi npm 2016

    decomp = normalize(decomp)

    lines = plt.plot(eigval, decomp, 'ro')
    plt.title('Mode spectrum for ' + title_label)
    # decomp_ax.set_yscale('log')
    plt.ylim(10**-6, 1.5)
    plt.xlim(1.1*min(eigval), 1.1*max(eigval))

    l = lines.pop(0)
    wl = weakref.ref(l)

    return [l]


def timestep_plot(current_pos, xy, Ni, Nk, BM, ax=None, factor=1, amp=1, title='', color_particles='k',
                  scat_cmap='gray', bondcolor=None, fontsize='auto', bgcolor='#d9d9d9', cmap='seismic', linewidth=2,
                  ptsize=-1.0, xlim=None, ylim=None, circ_edgecolor='k', circ_linewidth=1, suptitle='',
                  ticks_off=True, alpha=0.6, cmap_excite='isolum_rainbow', show_bonds=True, boundary=None):
    """makes plot in position space for simulation time step

    Parameters
    ----------
    current_pos: array 2nx3
        current positions of the gyros
    xy: array 2nx3
        Equilibrium position of the gyroscopes
    Ni : matrix of dimension n x (max number of neighbors)
            Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes
    Nk : matrix of dimension n x (max number of neighbors)
            Correponds to Ni matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection
    BM: NP x NN array
        rest bond lenth matrix, as NP x NN array
    ax : matplotlib axis
        axis on which to draw the network
    factor : float (optional):
        factor to multiply displacements by for drawing ( to see them better)
    amp : float(optional):
        colorlimit value is +/- amp. If amp == 'auto', sets limit as maximum stretch
    title : string
        Title of the figure
    color_particles: string ('k', 'angles')
        Whether to color the particles black ('k'), color them by their angle phi ('angles'), color by a vector
        with scat_cmap (supplied numpy array), color by a float with scat_cmap,
        ('mimic_expt') color them with black circles and white,
        ('mimic_expt_boundarypost') color them with black circles and white but plot the boundary as posts
        dots mimicking experiment, or don't color them (else)
    scat_cmap : colormap instance or string specifier for colormap
        The colormap to use if color_particles is a numpy array to color particles with a scatterplot
    bondcolor : None or color spec
        Color for the bonds
    fontsize : int or 'auto'
        fontsize
    bgcolor : color spec
        background color
    cmap : matplotlib colormap spec
        colormap for coloring bonds by strain
    linewidth : int
        width of the bonds
    ptsize : float
        size of the points to draw. If == -1, then uses min(max(0.01, 50./NP), 0.05)
    xlim : None or float
        Specify to enforce x limits of the axis
    ylim : None or float
        Specify to enforce y limits on the axis
    circ_edgecolor : string specifier (default='k')
        Color of line around each displacement circle at each point
    circ_linewidth : 1
        Linewidth of the displacement circle drawn around each point
    suptitle : ''
    ticks_off : bool (default=True)
        Turn off ticks on timestep plot axis
    alpha : opacity for displacement circles

    Returns
    ----------
    [scat_fg, p, lines_st]:
        things to be cleared before next time step is drawn
        """
    NP = len(xy)
    current_pos = np.reshape(current_pos, np.shape(xy))
    # We rename xy as R_p, so that we confusingly can redefine xy later
    R_p = xy
    # plt.subplots_adjust(left=0.00, right=1.0)

    if ax is None:
        ax = plt.gca()

    ax.set_axis_bgcolor(bgcolor)  # [214.5/255.,214.5/255.,214.5/255.] ) #'#E8E8E8')
    Rx = xy[:, 0]
    Ry = xy[:, 1]

    # Define the difference between current and rest positions
    diffs = current_pos - xy
    diffx = diffs[:, 0]
    diffy = diffs[:, 1]
    mag = np.sqrt(diffx**2 + diffy**2)
    angles = np.mod(np.arctan2(diffy, diffx), 2.*np.pi)

    # the displayed points
    scat_x = Rx + factor*diffx
    scat_y = Ry + factor*diffy

    # the actual points
    ss_x = Rx + diffx
    ss_y = Ry + diffy

    patch = [Circle((Rx[i], Ry[i]), radius=factor * mag[i]) for i in range(len(Rx))]

    z = np.zeros(len(scat_x))

    # Now, confusingly, R signifies the displayed points
    R = np.array([scat_x, scat_y, z]).T

    # norm for normal --> actual pts
    Rnorm = np.array([ss_x, ss_y, z]).T

    # Initialize streches vector to be longer than necessary
    inc = 0
    stretches = np.zeros(3 * len(R))

    for i in range(len(R)):
        # for j, k in zip(Ni[i], Nk[i]):
        for j, k, q in zip(Ni[i], Nk[i], BM[i]):
            if i < j and abs(k) > 0:
                # the distance between the actual points
                n1 = float(np.linalg.norm(Rnorm[i]-Rnorm[j]))

                # the distance between the rest points
                # n2 = np.linalg.norm(R_p[i] - R_p[j])
                # print 'n1 = ', n1
                # print 'BM[i] = ', BM[i]
                # stretches[inc] = (n1 - n2)
                stretches[inc] = n1 - q
                inc += 1

    test = list(np.zeros([inc, 1]))
    inc = 0
    for i in range(len(R)):
        for j, k in zip(Ni[i], Nk[i]):
            if i < j and abs(k) > 0:
                test[inc] = [R[(i, j), 0], R[(i, j), 1]]
                inc += 1

    if show_bonds:
        lines = [zip(x, y) for x, y in test]
        stretch = np.array(stretches[0:inc])
        # print 'stretch = ', stretch

        if bondcolor is None:
            lines_st = LineCollection(lines, array=stretch, cmap=cmap, linewidth=linewidth)
            if amp == 'auto':
                climv = max(abs(stretch))
                lines_st.set_clim([-climv, climv])
            else:
                # Enforce the coloring of bonds
                lines_st.set_clim([-amp, amp])
        else:
            lines_st = LineCollection(lines, color=bondcolor, linewidth=linewidth)

        lines_st.set_zorder(0)
        # print 'lines = ', lines
        ax.add_collection(lines_st)
    else:
        lines_st = []

    try:
        cmap_excite = plt.get_cmap(cmap_excite)
    except:
        lecmaps.register_colormaps([cmap_excite])

    p = PatchCollection(patch, cmap=cmap_excite, alpha=alpha, edgecolor=circ_edgecolor, linewidth=circ_linewidth)

    p.set_array(np.array(angles))
    p.set_clim([0, 2*np.pi])
    p.set_zorder(100)

    ax.add_collection(p)

    s = absolute_sizer()
    if ptsize == -1.0:
        ptsize = min(max(0.0005, 2./np.sqrt(float(NP))), 0.005)

    if color_particles == 'k':
        if NP < 10000:
            scat_fg = ax.scatter(scat_x, scat_y, s=s(ptsize), c='k', alpha=1.0)  # , zorder=200)
        else:
            scat_fg = None
    elif color_particles == 'angles':
        if NP > 1000:
            scat_fg = ax.scatter(scat_x, scat_y, s=s(ptsize), c=angles,
                                 vmin=0., vmax=2.*np.pi, cmap='isolum_rainbow',
                                 alpha=0.4, zorder=200)
        else:
            scat_fg = ax.scatter(scat_x, scat_y, s=s(ptsize), c=angles,
                                 vmin=0., vmax=2.*np.pi, cmap='isolum_rainbow', edgecolors='k',
                                 alpha=0.4, zorder=200)
    elif color_particles == 'mimic_expt':
        # Check if in virtualenv or not
        if hasattr(sys, 'real_prefix'):
            # not in virtual env
            blacksz = s(ptsize * 5)
            whitesz = s(ptsize * 1.5)
            lwsz = ptsize * 2
        else:
            # This seems almost completely random....
            blacksz = s(ptsize * 10)
            whitesz = s(ptsize * 0.4)
            lwsz = ptsize * 4

        scat_fg0 = ax.scatter(scat_x, scat_y, s=blacksz, facecolor='k', edgecolors='w', lw=lwsz, alpha=1.0, zorder=70)
        scat_fg1 = ax.scatter(scat_x, scat_y, s=whitesz, c='w', alpha=1.0, zorder=80)
        scat_fg = [scat_fg0, scat_fg1]
    elif color_particles == 'mimic_expt_boundarypost':
        # Check if in virtualenv or not
        if hasattr(sys, 'real_prefix'):
            # not in virtual env
            blacksz = s(ptsize * 5)
            whitesz = s(ptsize * 1.5)
            lwsz = ptsize * 2
        else:
            # This seems almost completely random....
            blacksz = s(ptsize * 10)
            whitesz = s(ptsize * 0.4)
            lwsz = ptsize * 4

        if boundary is None:
            import lepm.lattice_elasticity as le
            BL = le.NL2BL(Ni, Nk)
            boundary = le.extract_boundary(xy, Ni, Nk, BL)

        bulkinds = np.setdiff1d(np.arange(len(xy)), boundary)

        scat_fg0 = ax.scatter(scat_x[bulkinds], scat_y[bulkinds], s=blacksz,
                              facecolor='k', edgecolors='w', lw=lwsz, alpha=1.0, zorder=70)
        scat_fg1 = ax.scatter(scat_x[bulkinds], scat_y[bulkinds], s=whitesz, c='w', alpha=1.0, zorder=80)
        scat_fg2 = ax.scatter(scat_x[boundary], scat_y[boundary], s=whitesz, c='k', alpha=1.0, zorder=60)
        scat_fg = [scat_fg0, scat_fg1, scat_fg2]

    elif isinstance(color_particles, np.ndarray) or isinstance(color_particles, float):
        scat_fg = ax.scatter(scat_x, scat_y, s=s(5*ptsize), c=color_particles, lw=0.6, edgecolor='k', cmap=scat_cmap,
                             vmin=0., vmax=1.0, zorder=999999)
    else:
        scat_fg = []

    # Set axis limits
    ax.axis('scaled')

    if xlim is not None:
        if isinstance(xlim, float):
            ax.set_xlim(-xlim, xlim)
        else:
            ax.set_xlim(xlim)
    if ylim is not None:
        if isinstance(ylim, float):
            ax.set_ylim(-ylim, ylim)
        else:
            ax.set_ylim(ylim)

    if ticks_off:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # # NPM version: add colorbar here
    # fig = plt.gcf()
    # axcb = fig.colorbar(lines_st)
    # axcb.set_label('Strain')
    # axcb.set_clim(vmin=-climv,vmax=climv)

    # Get proper position for timestamp, title, etc
    if fontsize != 'auto':
        ax.set_title(title, fontsize=fontsize, y=1.01)
        if suptitle != '' and suptitle is not None:
            fig = plt.gcf()
            plt.text(0.5, 0.95, suptitle, transform=fig.transFigure, fontsize=fontsize, ha='center', va='center')
    else:
        ax.set_title(title)
        if suptitle != '' and suptitle is not None:
            fig = plt.gcf()
            plt.text(0.5, 0.95, suptitle, transform=fig.transFigure, ha='center', va='center')

    return [scat_fg, lines_st, p]


def timestep_plot_original(current_pos, R, Ni, Nk, ax, factor=1, amp=1, dist=0):
    """makes plot in position space for time step

    Parameters
    ----------
    current_pos: array 2nx3
        current positions of the gyros

    R: array 2nx3
        Equilibrium position of the gyroscopes

    Ni : matrix of dimension n x (max number of neighbors)
            Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes

    Nk : matrix of dimension n x (max number of neighbors)
            Correponds to Ni matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection

    ax:
        axis for the plot

    factor : float (optional):
        factor to multiply displacements by for drawing ( to see them better)

    amp : float(optional):
        amplitude of maximum displacement


    Returns
    ----------

    [scat_fg, p, lines_st]:
        things to be cleared before next time step is drawn
        """
    NP = len(R)
    ax.set_axis_bgcolor('#d9d9d9')
    current_pos = np.reshape(current_pos, np.shape(R))
    R_p = R
    plt.sca(ax)
    Rx = R[:, 0]
    Ry = R[:, 1]
    angles = np.zeros(NP+2)

    diffs = current_pos - R
    scat_x = np.zeros(NP)
    scat_y = np.zeros(NP)

    ss_x = np.zeros(NP)
    ss_y = np.zeros(NP)

    patch = []
    for i in range(NP):
        diffx = diffs[i, 0]
        diffy = diffs[i, 1]
        mag = np.sqrt(diffx**2 + diffy**2)
        if mag == 0:
            mag = 1

        angles[i] = np.arccos(diffx/mag)

        if diffy < 0:
            angles[i] = 2 * np.pi - angles[i]

        scat_x[i] = Rx[i] + factor * diffx
        scat_y[i] = Ry[i] + factor * diffy

        ss_x[i] = Rx[i] + diffx
        ss_y[i] = Ry[i] + diffy

        mag = np.sqrt(diffx**2 + diffy**2)
        circ = Circle((Rx[i], Ry[i]), radius=factor * mag)
        patch.append(circ)

    z = np.zeros(len(scat_x))
    R = np.array([scat_x, scat_y, z]).T
    Rnorm = np.array([ss_x, ss_y, z]).T

    inc = 0

    stretches = np.zeros(3 * len(R))
    for i in range(len(R)):
        for j, k in zip(Ni[i], Nk[i]):
            if i < j and abs(k) > 0:
                n1 = float(np.linalg.norm(Rnorm[i]-Rnorm[j]))
                n2 = np.linalg.norm(R_p[i] - R_p[j])
                stretches[inc] = (n1 - n2)
                inc += 1

    test = list(np.zeros([inc, 1]))

    inc = 0

    for i in range(len(R)):
        for j, k in zip(Ni[i], Nk[i]):
            if i < j and abs(k) > 0:
                test[inc] = [R[(i, j), 0], R[(i, j), 1]]
                inc += 1

    lines = [zip(x, y) for x, y in test]
    stretch = np.array(stretches[0:inc])

    lines_st = LineCollection(lines, array=stretch, cmap='seismic', linewidth=4)
    lines_st.set_clim([-1. * amp, 1 * amp])
    lines_st.set_zorder(0)

    p = PatchCollection(patch, cmap='hsv', alpha=0.6)
    p.set_array(np.array(angles))

    plt.set_clim([0, 2 * np.pi])
    plt.set_zorder(1)

    ax.add_collection(lines_st)
    ax.add_collection(p)

    ax.set_aspect('equal')
    s = absolute_sizer()
    scat_fg = ax.scatter(scat_x, scat_y, s=s(0.05), c='k', alpha=1, zorder=2)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.title(dist)

    return [scat_fg, lines_st, p]


def construct_timestep_and_decomp_plot(fig, pos_ax, decomp_ax, time_step, R, eigval, eigvect, Ni, Nk, factor, amp,
                                       title_label=''):
    plt.sca(pos_ax)
    Rx = R[:, 0]
    Ry = R[:, 1]
    plt.xlim(Rx.min()-1, Rx.max()+1)
    plt.ylim(Ry.min()-1, Ry.max()+1)
    pos_ax.set_autoscale_on(False)
    s = absolute_sizer()
    ppu = get_points_per_unit()
    # v1 = decomp_plot(fig, time_step-R, eigvect, eigval, title_label, decomp_ax)
    v = timestep_plot(time_step, R, Ni, Nk, pos_ax, factor, amp, dist=title_label)

    return fig, v


def initialize_timestep_and_decomp_plot():
    """initializes axes for the eigenvalue/density of states plot.  calls functions to draw lattice and DOS

    Parameters
    ----------

    Returns
    ----------
    fig :
        figure with lattice and DOS drawn
    pos_ax:
        axis for the position plot

    decomp_ax
        axis for decomposition plot
        """
    # fig = plt.figure(figsize = (7.5, 10))
    # pos_ax = plt.axes([0.1, 0.35, 0.8, 0.6])  #axes constructor axes([left, bottom, width, height])
    # decomp_ax = plt.axes([0.1, 0.05, 0.8, 0.25])

    fig = plt.figure(figsize=(15, 15))
    pos_ax = plt.axes([0.1, 0.1, 0.8, 0.8])  # axes constructor axes([left, bottom, width, height])

    return fig, pos_ax


def get_points_per_unit(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.apply_aspect()
    x0, x1 = ax.get_xlim()
    return ax.bbox.width / abs(x1 - x0)


def absolute_sizer(ax=None):
    ppu = get_points_per_unit(ax)
    return lambda x: np.pi * (x*ppu)**2


def normalize(vector):
    N = len(vector)

    tot = 0

    for i in range(N):
        tot += abs(vector[i])**2

    return vector / np.sqrt(tot)


def initialize_1panel_centered_fig(Wfig=90, Hfig=None, wsfrac=0.4, hsfrac=None, vspace=8, hspace=8,
                                   tspace=10, fontsize=8, evenpix=True, dpi=150):
    """Creates a plot with one axis in the center of the canvas.

    Parameters
    ----------
    Wfig : width of the figure in mm
    Hfig : float or None
        height of the figure in mm. If None, uses Hfig = y0 + hs + vspace + hscbar + tspace
    x0frac : fraction of Wfig to leave blank left of plot
    y0frac : fraction of Wfig to leave blank below plot
    wsfrac : fraction of Wfig to make width of subplot
    hs : height of subplot in mm. If none, uses ws = wsfrac * Wfig
    vspace : vertical space between subplots
    hspace : horizontal space btwn subplots
    tspace : space above top figure
    fontsize : size of text labels, title

    Returns
    -------
    fig
    ax
    """
    # Make figure
    x0 = Wfig * (1 - wsfrac) * 0.5
    if Hfig is None:
        Hfig = Wfig * 0.75
    ws = round(Wfig * wsfrac)
    if hsfrac is None:
        hs = ws
    else:
        hs = Wfig * hsfrac
    y0 = (Hfig - hs - tspace) * 0.5
    if evenpix:
        wpix = Wfig / 25.4 * dpi
        hpix = Hfig / 25.4 * dpi
        fig = sps.figure_in_pixels(wpix, hpix, dpi=dpi)
    else:
        fig = sps.figure_in_mm(Wfig, Hfig)
    label_params = dict(size=fontsize, fontweight='normal')
    ax = sps.axes_in_mm((Wfig - ws) * 0.5, y0, ws, hs, label='', label_params=label_params)
    return fig, ax


def initialize_1panel_fig(Wfig=90, Hfig=None, x0frac=None, y0frac=0.1, wsfrac=0.4, hsfrac=None, vspace=8, hspace=8,
                          tspace=10, fontsize=8):
    """Creates a plot with one axis in the center of the canvas.

    Parameters
    ----------
    Wfig : width of the figure in mm
    Hfig : float or None
        height of the figure in mm. If None, uses Hfig = y0 + hs + vspace + hscbar + tspace
    x0frac : fraction of Wfig to leave blank left of plot
    y0frac : fraction of Wfig to leave blank below plot
    wsfrac : fraction of Wfig to make width of subplot
    hs : height of subplot in mm. If none, uses ws = wsfrac * Wfig
    vspace : vertical space between subplots
    hspace : horizontal space btwn subplots
    tspace : space above top figure
    fontsize : size of text labels, title

    Returns
    -------
    fig
    ax
    """
    # Make figure
    y0 = round(Wfig * y0frac)
    ws = round(Wfig * wsfrac)
    if hsfrac is None:
        hs = ws
    else:
        hs = hsfrac * Wfig
    if Hfig is None:
        Hfig = y0 + hs + tspace
    if x0frac is None:
        x0 = (Wfig - ws) * 0.5
    else:
        x0 = round(Wfig * x0frac)
    fig = sps.figure_in_mm(Wfig, Hfig)
    label_params = dict(size=fontsize, fontweight='normal')
    ax = sps.axes_in_mm(x0, y0, ws, hs, label='', label_params=label_params)
    return fig, ax


def initialize_1panel_cbar_fig(Wfig=90, Hfig=None, x0frac=0.15, y0frac=0.1, wsfrac=0.4, hsfrac=None,
                               wcbarfrac=0.05, hcbarfrac=0.7,
                               vspace=8, hspace=5, tspace=10, fontsize=8):
    """Creates a plot with one axis in the center of the canvas and a horizontal colorbar above it.

    Parameters
    ----------
    Wfig : width of the figure in mm
    Hfig : float or None
        height of the figure in mm. If None, uses Hfig = y0 + hs + vspace + hscbar + tspace
    x0frac : fraction of Wfig to leave blank left of plot
    y0frac : fraction of Wfig to leave blank below plot
    wsfrac : fraction of Wfig to make width of subplot
    hs : float or None
        height of subplot in mm. If None, uses ws = wsfrac * Wfig
    wcbarfrac : float
        width of the colorbar as fraction of the panel width (ws)
    hcbarfrac : float
        height of the colorbar as fraction of the panel height (hs)
    vspace : vertical space between subplots
    hspace : horizontal space btwn panel and cbar
    tspace : space above top figure
    fontsize : size of text labels, title

    Returns
    -------
    fig
    ax
    """
    # Make figure
    x0 = round(Wfig * x0frac)
    y0 = round(Wfig * y0frac)
    ws = round(Wfig * wsfrac)
    if hsfrac is None:
        hs = ws
    else:
        hs = hsfrac * Wfig
    wcbar = wcbarfrac * ws
    hcbar = hcbarfrac * hs
    if Hfig is None:
        Hfig = y0 + hs + tspace

    fig = sps.figure_in_mm(Wfig, Hfig)
    label_params = dict(size=fontsize, fontweight='normal')
    ax = sps.axes_in_mm((Wfig - ws) * 0.5, y0, ws, hs, label='', label_params=label_params)
    cbar_ax = sps.axes_in_mm((Wfig + ws) * 0.5 + hspace, y0 + (1. - hcbarfrac) * hs * 0.5, wcbar, hcbar,
                             label='', label_params=label_params)
    return fig, ax, cbar_ax


def initialize_nxmpanel_cbar_fig(nn, mm, Wfig=90, Hfig=None, x0frac=0.15, y0frac=0.1, wsfrac=0.2, hsfrac=None,
                                 wcbarfrac=0.05, hcbarfrac=0.7,
                                 vspace=8, hspace=5, tspace=10, fontsize=8, x0cbarfrac=None, y0cbarfrac=None,
                                 orientation='vertical', cbar_placement='default'):
    """Creates a plot with N x M axes grid (in horizontal row) and a colorbar.

    Parameters
    ----------
    nn : int
        Number of rows of axes
    mm : int
        Number of cols of axes
    Wfig : float or int
        width of the figure in mm
    Hfig : float or None
        height of the figure in mm. If None, uses Hfig = y0 + hs + vspace + hscbar + tspace
    x0frac : float
        fraction of Wfig to leave blank left of plot
    y0frac : float
        fraction of Wfig to leave blank below plot
    wsfrac : float
        fraction of Wfig to make width of subplot
    hsfrac : float or None
        height of subplot in fraction of figure width. If None, uses hs = wsfrac * Wfig
    wcbarfrac : float
        width of the colorbar as fraction of the panel width (ws)
    hcbarfrac : float
        height of the colorbar as fraction of the panel height (hs)
    vspace : float or int
        vertical space between subplots
    hspace : float or int
        horizontal space btwn panel and cbar
    tspace : float or int
        space above top figure in mm
    fontsize : int
        size of text labels, title
    x0cbarfrac : float or None
    y0cbarfrac : float or None
    orientation : str
    cbar_placement : str (default='default')
        Description for placement for cbar x0 and y0, used if x0cbarfrac and/or y0cbarfrac is None
        ['above_center', 'right_right', 'above_right']

    Returns
    -------
    fig
    ax
    """
    # Make figure
    x0 = round(Wfig * x0frac)
    y0 = round(Wfig * y0frac)
    ws = round(Wfig * wsfrac)
    if hsfrac is None:
        hs = ws
    else:
        hs = hsfrac * Wfig
    wcbar = wcbarfrac * ws
    hcbar = hcbarfrac * hs
    if Hfig is None:
        Hfig = y0 + (hs + vspace) * nn + tspace

    fig = sps.figure_in_mm(Wfig, Hfig)
    label_params = dict(size=fontsize, fontweight='normal')
    ax = []
    for nii in range(nn):
        for mii in range(mm):
            ax.append(sps.axes_in_mm(x0 + (ws + hspace) * mii,
                                     y0 + (nn - nii - 1) * (hs + vspace), ws, hs, label='', label_params=label_params))

    # Placement of the colorbar.
    # Note: to put colorbar over right subplot:
    # set x0cbarfrac = (x0frac + (mm - 0.5) * (wsfrac + hspace / Wfig) - wcbarfrac * 0.5)
    if x0cbarfrac is None:
        if orientation == 'vertical':
            if cbar_placement in ['right_right', 'default']:
                x0cbar = x0 + mm * ws + hspace * mm
            else:
                # todo: add cases here
                x0cbar = x0 + mm * ws + hspace * mm
        elif orientation == 'horizontal':
            if cbar_placement in ['above_center']:
                x0cbar = (Wfig - wcbar) * 0.5
            elif cbar_placement in ['above_right', 'default']:
                print 'mm = ', mm
                x0cbar = x0 + (mm - 0.5) * ws - wcbar * 0.5 + (mm - 1) * hspace
    else:
        x0cbar = x0cbarfrac * Wfig

    print 'leplt.nxm: orientation = ', orientation, '\ncbar_placement= ', cbar_placement
    if y0cbarfrac is None:
        if orientation == 'vertical':
            if cbar_placement in ['right_right', 'default']:
                y0cbar = y0 + (1. - hcbarfrac) * hs * 0.5
            else:
                y0cbar = y0 + (1. - hcbarfrac) * hs * 0.5
        elif orientation == 'horizontal':
            if cbar_placement in ['above_right', 'default']:
                print 'nn = ', nn
                y0cbar = y0 + (nn - 1.) * vspace + (nn + 0.1) * hs
            elif cbar_placement in ['above_center']:
                print 'leplt: placing cbar in center above subplots...'
                y0cbar = y0 + (nn - 1.) * vspace + (nn + 0.1) * hs
            else:
                y0cbar = y0 + (nn - 1.) * vspace + (nn + 0.1) * hs
    else:
        y0cbar = y0cbarfrac * Hfig
    cbar_ax = sps.axes_in_mm(x0cbar, y0cbar, wcbar, hcbar, label='', label_params=label_params)

    return fig, ax, cbar_ax


def initialize_nxmpanel_fig(nn, mm, Wfig=90, Hfig=None, x0frac=0.15, y0frac=0.1, wsfrac=0.2, hsfrac=None,
                            wcbarfrac=0.05, hcbarfrac=0.7, vspace=8, hspace=5,
                            tspace=10, fontsize=8):
    """Creates a plot with N x M axes grid (in horizontal row)

    Parameters
    ----------
    Wfig : width of the figure in mm
    Hfig : float or None
        height of the figure in mm. If None, uses Hfig = y0 + hs + vspace + hscbar + tspace
    x0frac : fraction of Wfig to leave blank left of plot
    y0frac : fraction of Wfig to leave blank below plot
    wsfrac : fraction of Wfig to make width of subplot
    hsfrac : float or None
        height of subplot in units of Wfig. If None, uses hs = ws = wsfrac * Wfig
    wcbarfrac : float
        width of the colorbar as fraction of the panel width (ws)
    hcbarfrac : float
        height of the colorbar as fraction of the panel height (hs)
    vspace : vertical space between subplots
    hspace : horizontal space btwn panel and cbar
    tspace : space above top figure
    fontsize : size of text labels, title

    Returns
    -------
    fig
    ax
    """
    # Make figure
    x0 = round(Wfig * x0frac)
    y0 = round(Wfig * y0frac)
    ws = round(Wfig * wsfrac)
    if hsfrac is None:
        hs = ws
    else:
        hs = hsfrac * Wfig
    wcbar = wcbarfrac * ws
    hcbar = hcbarfrac * hs
    if Hfig is None:
        Hfig = y0 + hs * nn + vspace * (nn - 1) + tspace

    fig = sps.figure_in_mm(Wfig, Hfig)
    label_params = dict(size=fontsize, fontweight='normal')
    ax = []
    for nii in range(nn):
        for mii in range(mm):
            ax.append(sps.axes_in_mm(x0 + (ws + hspace) * mii,
                                     y0 + (nn - nii - 1) * (hs + vspace), ws, hs, label='', label_params=label_params))

    return fig, ax


def initialize_1panel_cbar_cent(Wfig=90, Hfig=None, wsfrac=0.4, hsfrac=None, cbar_pos='above',
                                wcbarfrac=0.6, hcbarfrac=0.05, cbar_label='',
                                vspace=8, hspace=5, tspace=10, fontsize=8, evenpix=True, dpi=150):
    """Creates a plot with one axis instance in the center of the canvas, with colorbar above the axis.

    Parameters
    ----------
    Wfig : width of the figure in mm
    Hfig : float or None
        height of the figure in mm. If None, uses Hfig = y0 + hs + vspace + hscbar + tspace
    wsfrac : float
        fraction of Wfig to make width of subplot
    hsfrac : float or None
        fraction of Wfig to make height of subplot. If None, uses hs = wsfrac * Wfig
    cbar_pos : str specifier ('above', 'right')
        Where to place the colorbar
    wcbarfrac : float
        width of the colorbar as fraction of the panel width (ws)
    hcbarfrac : float
        height of the colorbar as fraction of the panel height (hs)
    vspace : vertical space between subplots
    hspace : horizontal space btwn panel and cbar
    tspace : space above top figure
    fontsize : size of text labels, title

    Returns
    -------
    fig
    ax
    """
    # Make figure
    if Hfig is None:
        Hfig = Wfig * 0.75
    ws = round(Wfig * wsfrac)
    if hsfrac is None:
        hs = ws
    else:
        hs = Wfig * hsfrac
    y0 = (Hfig - hs - tspace) * 0.5
    wcbar = wcbarfrac * ws
    hcbar = hcbarfrac * hs
    if evenpix:
        wpix = Wfig / 25.4 * dpi
        hpix = Hfig / 25.4 * dpi
        fig = sps.figure_in_pixels(wpix, hpix, dpi=dpi)
    else:
        fig = sps.figure_in_mm(Wfig, Hfig)

    label_params = dict(size=fontsize, fontweight='normal')
    ax = sps.axes_in_mm((Wfig - ws) * 0.5, y0, ws, hs, label='', label_params=label_params)
    if cbar_pos == 'right':
        cbar_ax = sps.axes_in_mm((Wfig + ws) * 0.5 + hspace, y0 + (1. - hcbarfrac) * hs * 0.5, wcbar, hcbar,
                                 label=cbar_label, label_params=label_params)
    elif cbar_pos == 'above':
        cbar_ax = sps.axes_in_mm((Wfig - wcbar) * 0.5, y0 + hs + vspace, wcbar, hcbar,
                                 label=cbar_label, label_params=label_params)
    return fig, ax, cbar_ax


def initialize_2panel_3o4ar_cent(Wfig=360, Hfig=270, fontsize=8, wsfrac=0.4, wssfrac=0.3, x0frac=0.1, y0frac=0.1):
    """Returns 2 panel figure with left axis square and right axis 3/4 aspect ratio

    Returns
    -------
    fig :
    ax :
    """
    fig = sps.figure_in_mm(Wfig, Hfig)
    ws = wsfrac * Wfig
    hs = ws
    wss = wssfrac * Wfig
    hss = wss * 3. / 4.
    x0 = x0frac * Wfig
    y0 = y0frac * Wfig
    label_params = dict(size=fontsize, fontweight='normal')
    ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
          for x0, y0, width, height, part in (
              [x0, (Hfig - hs) * 0.5, ws, hs, ''],  # network and kitaev regions
              [Wfig - wss - x0, (Hfig - hss) * 0.5, wss, hss, '']  # plot for chern
          )]
    return fig, ax


def initialize_2panel_cbar_cent(Wfig=360, Hfig=270, fontsize=12, wsfrac=0.4, wssfrac=0.3, x0frac=0.1, y0frac=0.1,
                                wcbarfrac=0.15, hcbar_fracw=0.1, vspace=5):
    """Returns 2 panel figure with left axis square and right axis either square or 3/4 aspect ratio, depending on
     if right3o4 is True,  but also 2 colorbar axes (one for each panel).
     The colorbars are above the plot axes by default.

    Parameters
    ----------
    hcbar_fracw : float
        height of the colorbar, as a fraction of the width of the colorbar

    Returns
    -------
    fig :
    ax :
    cbar_ax:
    """
    fig = sps.figure_in_mm(Wfig, Hfig)
    ws = wsfrac * Wfig
    hs = ws
    wss = wssfrac * Wfig
    hss = wss * 3. / 4.
    wcbar = wcbarfrac * Wfig
    hcbar = hcbar_fracw * wcbar
    x0 = x0frac * Wfig
    y0 = y0frac * Wfig
    label_params = dict(size=fontsize, fontweight='normal')
    ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
          for x0, y0, width, height, part in (
              [x0, (Hfig - hs) * 0.5, ws, hs, ''],  # network and kitaev regions
              [Wfig - wss - x0, (Hfig - hss) * 0.5, wss, hss, '']  # plot for chern
          )]
    cbar_ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
               for x0, y0, width, height, part in (
                   [x0 + (ws - wcbar)*0.5, (Hfig + hs) * 0.5 + vspace, wcbar, hcbar, ''],  # left cbar above
                   [Wfig - wss - x0 + (wss - wcbar)*0.5, (Hfig + hss) * 0.5 + vspace, wcbar, hcbar, '']  # right cbar
               )]
    return fig, ax, cbar_ax


def initialize_axis_stack(n_ax, make_cbar=False, Wfig=90, Hfig=90, hfrac=None, wfrac=0.6, x0frac=None, y0frac=0.12,
                          vspace=5, hspace=5, fontsize=8, wcbar_frac=0.2, cbar_aratio=0.1, cbar_orientation='vertical',
                          cbarspace=5, tspace=8):
    """Create a vertical stack of plots, and a colorbar if make_cbar is True

    Parameters
    ----------
    n_ax : int
        number of axes to draw
    make_cbar : bool
        Create a colorbar on the figure
    Wfig : int or float
        width of figure in mm
    Hfig : int or float
        height of figure in mm
    hfrac : float or None
        Fraction of Hfig to make each axis height (hs)
    wfrac : float
        Fraction of Wfig to make each axis width (ws)
    x0frac : float or None
        Buffer room in mm on the left of all the axes in the stack. If None, centers the axes.
    y0frac : float or None
        Buffer room in mm on the bottom of the lowest axis in the stack
    vspace : float or int
        vertical space in mm between each axis in the stack
    hspace : float or int
        space between the stack of axes and the colorbar, in mm
    fontsize : int
        font size for axis params
    hcbar_frac : float
        fraction of Wfig to make height of colorbar
    cbar_orientation : str ('vertical', 'horizontal')
        Orientation of the colorbar

    """
    # This method returns and ImageGrid instance
    # ax = AxesGrid(fig, 111,  # similar to subplot(111)
    #               nrows_ncols=(n_ax, 1),  # creates 2x2 grid of axes
    #               axes_pad=0.1,  # pad between axes in inch.
    #               share_all=True,
    #               )
    fig = sps.figure_in_mm(Wfig, Hfig)
    label_params = dict(size=fontsize, fontweight='normal')

    if hfrac is None:
        hfrac = 0.8 / float(n_ax) - ((n_ax - 2.) * float(vspace) + tspace) / (float(Hfig) * float(n_ax))
        if make_cbar and cbar_orientation == 'horizontal':
            # colorbar is going on top, with space cbarspace
            hfrac -= float(cbarspace) / (float(Hfig) * float(n_ax))
        print 'hfrac = ', hfrac
    if x0frac is None:
        x0 = (1. - wfrac)*0.5 * Wfig
    else:
        x0 = x0frac * Wfig
    y0 = y0frac * Hfig
    ws = wfrac * Wfig
    hs = hfrac * Hfig
    print 'hs = ', hs
    xywh_list = [[x0, y0 + (n_ax - 1 - ii) * (hs + vspace), ws, hs, ''] for ii in range(n_ax)]

    print 'xywh_list = ', xywh_list
    ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
          for x0, y0, width, height, part in xywh_list]

    if make_cbar:
        wcbar = Wfig * wcbar_frac
        hcbar = cbar_aratio * wcbar
        if cbar_orientation == 'vertical':
            cbar_ax = sps.axes_in_mm(x0 + ws + hspace, (Hfig - wcbar) * 0.5, hcbar, wcbar, label='',
                                     label_params=label_params)
        elif cbar_orientation == 'horizontal':
            cbar_ax = sps.axes_in_mm(x0 + (ws - wcbar) * 0.5, y0 + n_ax * (hs + vspace) + cbarspace, wcbar, hcbar,
                                     label='', label_params=label_params)
    else:
        cbar_ax = None

    return fig, ax, cbar_ax


def initialize_axis_doublestack(n_ax, make_cbar=False, Wfig=90, Hfig=90, hfrac=None, wfrac=0.3, x0frac=None,
                                y0frac=0.12, vspace=5, hspace=5, fontsize=8, wcbar_frac=0.2, cbar_aratio=0.1,
                                cbarspace=5, tspace=8):
    """Create a vertical stack of plots, and a colorbar if make_cbar is True

    Parameters
    ----------
    n_ax : int
        number of axes to draw
    make_cbar : bool
        Create a colorbar on the figure (will be horizontal, above subplots)
    Wfig : int or float
        width of figure in mm
    Hfig : int or float
        height of figure in mm
    hfrac : float or None
        Fraction of Hfig to make each axis height (hs)
    wfrac : float
        Fraction of Wfig to make each axis width (ws)
    x0frac : float or None
        Buffer room in mm on the left of all the axes in the stack. If None, centers the axes.
    y0frac : float or None
        Buffer room in mm on the bottom of the lowest axis in the stack
    vspace : float or int
        vertical space in mm between each axis in the stack
    hspace : float or int
        space between the stack of axes and the colorbar, in mm
    fontsize : int
        font size for axis params
    hcbar_frac : float
        fraction of Wfig to make height of colorbar

    Returns
    -------
    fig : matplotlib figure instance
    ax : list of matplotlib axis instances
    cbar_ax : list of 2 matplotlib axis instances

    """
    # This method returns and ImageGrid instance
    # ax = AxesGrid(fig, 111,  # similar to subplot(111)
    #               nrows_ncols=(n_ax, 1),  # creates 2x2 grid of axes
    #               axes_pad=0.1,  # pad between axes in inch.
    #               share_all=True,
    #               )
    fig = sps.figure_in_mm(Wfig, Hfig)
    label_params = dict(size=fontsize, fontweight='normal')

    if hfrac is None:
        hfrac = 0.8 / float(n_ax) - ((n_ax - 2.) * float(vspace) + tspace) / (float(Hfig) * float(n_ax))
        if make_cbar:
            # colorbar is going on top, with space cbarspace
            hfrac -= float(cbarspace) / (float(Hfig) * float(n_ax))
        print 'hfrac = ', hfrac
    if x0frac is None:
        x0 = (1. - 2.*wfrac - float(hspace)/float(Wfig))*0.5 * Wfig
    else:
        x0 = x0frac * Wfig
    print 'x0 = ', x0
    y0 = y0frac * Hfig
    ws = wfrac * Wfig
    hs = hfrac * Hfig
    xywh_list = [[x0, y0 + (n_ax - 1 - ii) * (hs + vspace), ws, hs, ''] for ii in range(n_ax)]
    xywh2_list = [[x0 + ws + hspace, y0 + (n_ax - 1 - ii) * (hs + vspace), ws, hs, ''] for ii in range(n_ax)]

    print 'xywh_list = ', xywh_list
    ax = [sps.axes_in_mm(x, y, width, height, label=part, label_params=label_params)
          for x, y, width, height, part in xywh_list]
    ax += [sps.axes_in_mm(x, y, width, height, label=part, label_params=label_params)
           for x, y, width, height, part in xywh2_list]
    if make_cbar:
        wcbar = Wfig * wcbar_frac
        hcbar = cbar_aratio * wcbar
        print 'wcbar = ', wcbar
        print 'x0 = ', x0
        print 'ws = ', ws
        cbar_ax = [sps.axes_in_mm(x0 + (ws - wcbar) * 0.5, y0 + n_ax * (hs + vspace) + cbarspace, wcbar,
                                  hcbar, label='', label_params=label_params),
                   sps.axes_in_mm(x0 + (3.*ws - wcbar) * 0.5 + hspace, y0 + n_ax * (hs + vspace) + cbarspace, wcbar,
                                  hcbar, label='', label_params=label_params)]
    else:
        cbar_ax = None

    return fig, ax, cbar_ax


def initialize_insetaxis_doublestack(n_ax, make_cbar=False, Wfig=90, Hfig=90, hfrac=None, wfrac=0.3, x0frac=None,
                                     y0frac=0.12, vspace=5, hspace=5, fontsize=8, wcbar_frac=0.2, cbar_aratio=0.1,
                                     cbarspace=5, tspace=8, ins_pad=3, ins_pad_right=None, wins=None, hins=None):
    """Create a vertical stack of plots, and a colorbar if make_cbar is True

    Parameters
    ----------
    n_ax : int
        number of axes to draw
    make_cbar : bool
        Create a colorbar on the figure (will be horizontal, above subplots)
    Wfig : int or float
        width of figure in mm
    Hfig : int or float
        height of figure in mm
    hfrac : float or None
        Fraction of Hfig to make each axis height (hs)
    wfrac : float
        Fraction of Wfig to make each axis width (ws)
    x0frac : float or None
        Buffer room in mm on the left of all the axes in the stack. If None, centers the axes.
    y0frac : float or None
        Buffer room in mm on the bottom of the lowest axis in the stack
    vspace : float or int
        vertical space in mm between each axis in the stack
    hspace : float or int
        space between the stack of axes and the colorbar, in mm
    fontsize : int
        font size for axis params
    hcbar_frac : float
        fraction of Wfig to make height of colorbar

    Returns
    -------
    fig : matplotlib figure instance
    ax : list of matplotlib axis instances
    cbar_ax : list of 2 matplotlib axis instances

    """
    # This method returns and ImageGrid instance
    # ax = AxesGrid(fig, 111,  # similar to subplot(111)
    #               nrows_ncols=(n_ax, 1),  # creates 2x2 grid of axes
    #               axes_pad=0.1,  # pad between axes in inch.
    #               share_all=True,
    #               )
    fig = sps.figure_in_mm(Wfig, Hfig)
    label_params = dict(size=fontsize, fontweight='normal')

    if hfrac is None:
        hfrac = 0.8 / float(n_ax) - ((n_ax - 2.) * float(vspace) + tspace) / (float(Hfig) * float(n_ax))
        if make_cbar:
            # colorbar is going on top, with space cbarspace
            hfrac -= float(cbarspace) / (float(Hfig) * float(n_ax))
        print 'hfrac = ', hfrac
    if x0frac is None:
        x0 = (1. - 2.*wfrac - float(hspace)/float(Wfig))*0.5 * Wfig
    else:
        x0 = x0frac * Wfig
    print 'x0 = ', x0
    y0 = y0frac * Hfig
    ws = wfrac * Wfig
    hs = hfrac * Hfig
    xywh_list = [[x0, y0 + (n_ax - 1 - ii) * (hs + vspace), ws, hs, ''] for ii in range(n_ax)]
    xywh2_list = [[x0 + ws + hspace, y0 + (n_ax - 1 - ii) * (hs + vspace), ws, hs, ''] for ii in range(n_ax)]

    print 'xywh_list = ', xywh_list
    ax = [sps.axes_in_mm(x, y, width, height, label=part, label_params=label_params)
          for x, y, width, height, part in xywh_list]
    ax += [sps.axes_in_mm(x, y, width, height, label=part, label_params=label_params)
           for x, y, width, height, part in xywh2_list]
    if make_cbar:
        wcbar = Wfig * wcbar_frac
        hcbar = cbar_aratio * wcbar
        print 'wcbar = ', wcbar
        print 'x0 = ', x0
        print 'ws = ', ws
        cbar_ax = [sps.axes_in_mm(x0 + (ws - wcbar) * 0.5, y0 + n_ax * (hs + vspace) + cbarspace, wcbar,
                                  hcbar, label='', label_params=label_params),
                   sps.axes_in_mm(x0 + (3.*ws - wcbar) * 0.5 + hspace, y0 + n_ax * (hs + vspace) + cbarspace, wcbar,
                                  hcbar, label='', label_params=label_params)]
    else:
        cbar_ax = None

    if wins is None:
        wins = hs
    if hins is None:
        hins = hs
    if ins_pad_right is None:
        ins_pad_right = ins_pad
    x0ins = min(max(x0 - wins - ins_pad, 0), x0 * 0.5)
    xywh_list = [[x0ins, y0 + (n_ax - 1 - ii) * (hs + vspace) + (hs - hins)*0.5, wins, hins, ''] for ii in range(n_ax)]
    xywh2_list = [[x0 + ws*2 + hspace + ins_pad_right, y0 + (n_ax - 1 - ii) * (hs + vspace) + (hs - hins)*0.5,
                   wins, hins, ''] for ii in range(n_ax)]
    inset_ax = [sps.axes_in_mm(x, y, width, height, label=part, label_params=label_params)
                for x, y, width, height, part in xywh_list]
    inset_ax += [sps.axes_in_mm(x, y, width, height, label=part, label_params=label_params)
                 for x, y, width, height, part in xywh2_list]
    return fig, ax, cbar_ax, inset_ax


def cbar_ax_is_vertical(cbar_ax):
    """Determine if a colorbar axis is vertical or not (ie it is horizontal) based on its dimensions"""
    bbox = cbar_ax.get_window_extent()  # .transformed(fig.dpi_scale_trans.inverted())
    return bbox.width < bbox.height


def lt2description(lp):
    """
    Convert latticetopology string shorthand into a description for a title.
    """
    lt = lp['LatticeTop']
    if lt == 'hucentroid':
        return 'voronoized hyperuniform network'
    elif lt == 'kagome_hucent':
        return 'kagomized hyperuniform network'
    elif lt == 'kagper_hucent':
        return r'partially kagomized hyperuniform ($d=${0:0.2f}'.format(lp['percolation_density']) + ') network'
    elif lt == 'hexagonal':
        return 'honeycomb network'
    elif lt == 'iscentroid':
        return 'voronoized jammed network'
    elif lt == 'kagome_isocent':
        return 'kagomized jammed network'
    elif lt == 'penroserhombTricent':
        return 'voronoized rhombic Penrose lattice'
    elif lt == 'kagome_penroserhombTricent':
        return 'kagomized rhombic Penrose lattice'
    elif lt in ['hex_kagframe', 'hex_kagcframe']:
        return 'honeycomb lattice with kagome frame'