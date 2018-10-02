import numpy as np
import fenics_handling as pfh
import lepm.dataio as dio
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

##########################################
# Plotting
##########################################

def collect_lines(xy, BL, bs, climv):
    """Creates collection of line segments, colored according to an array of values.

    Parameters
    ----------
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    bs : array of dimension #bonds x 1
        Strain in each bond
    climv : float or tuple
        Color limit for coloring bonds by bs

    Returns
    ----------
    line_segments : matplotlib.collections.LineCollection
        Collection of line segments
    """
    lines = [zip(xy[BL[i, :], 0], xy[BL[i, :], 1]) for i in range(len(BL))]
    line_segments = LineCollection(lines,  # Make a sequence of x,y pairs
                                   linewidths=(1.),  # could iterate over list
                                   linestyles='solid',
                                   cmap='coolwarm',
                                   norm=plt.Normalize(vmin=-climv, vmax=climv))
    line_segments.set_array(bs)
    print(lines)
    return line_segments


def pf_mlabplot(X, Y, Z, C, fname, XYZboundary=np.array([]), SZ=800, elang=75, azang=45, cmppath=''):
    """Plot colored surface using mayavi.

    X,Y,Z : Nx1 arrays
        X,Y,Z positions of pts on surface
    C : Nx1 array
        values to map into color using colormap, one val for each point
    fname : string
        name to save image, if empty string, then simply displays plot instead of saving it
    XYZboundary : numpy Nx3 array, default is empty list
        boundary points, to be colored black, if not empty list
    SZ : int or list of 2 ints
        pixel width and height of square image (single int), or dimensions in pixels of image to print (list of ints)
    elang : float (default=75)
        elevation angle of camera
    azang : float (default=45)
        azimuth angle of camera
    Cmp : string filename (with path) to csv colormap (default is empty)
        if nonempty, use the csv file designated as a colormap using LUT
    """
    from mayavi import mlab
    # Define the points in 3D space
    # including color code based on Z coordinate.
    print 'Setting up figure...'
    mlab.close(all=True)
    fig = mlab.figure(size=(1600, 1200))
    pts = mlab.points3d(X, Y, Z, C)

    # Triangulate based on X, Y with Delaunay 2D algorithm.
    print 'Triangulating points...'
    mesh = mlab.pipeline.delaunay2d(pts)

    # Remove the point representation from the plot
    pts.remove()

    # Draw a surface based on the triangulation
    surf = mlab.pipeline.surface(mesh)

    if cmppath != '':
        Cmp = np.loadtxt(cmppath, delimiter=',')
        # CHANGE COLORBAR USING LUT
        lut = surf.module_manager.scalar_lut_manager.lut.table.to_array()

        # The lut is a 256x4 array, with the columns representing RGBA
        # (red, green, blue, alpha) coded with integers going from 0 to 255.
        # Make dims of Cmap match lut (not necessary if we change the number of colors)
        ntv = len(Cmp)
        temp = np.linspace(0, ntv - 1, ntv)
        inds = np.around(temp * len(Cmp) / ntv)  # indices of Cmp to sample
        Cmp2lut = Cmp[inds.astype(int), :] * 255.
        newlut = np.hstack((Cmp2lut, np.ones((len(Cmp2lut), 1,)) * 255.))

        # Replace lut with colormap
        # lut[:, 0:4] = newlut.astype(uint8)
        # and finally we put this LUT back in the surface object. We could have
        # added any 255*4 array rather than modifying an existing LUT.
        surf.module_manager.scalar_lut_manager.lut.table = newlut
        surf.module_manager.scalar_lut_manager.number_of_colors = ntv
        cmax = surf.module_manager.scalar_lut_manager.lut.table_range[1]
        surf.module_manager.scalar_lut_manager.lut.table_range = np.array([-cmax, cmax])

        inspect_it = surf.module_manager.scalar_lut_manager.lut
        print(inspect_it)
        ####

    # We need to force update of the figure now that we have changed the LUT.
    mlab.draw()

    if XYZboundary:
        xyzb = XYZboundary
        # Draw boundary contour
        mlab.plot3d(xyzb[:, 0], xyzb[:, 1], xyzb[:, 2], tube_radius=0.001, color=(0, 0, 0))

    # mlab.show()
    mlab.view(elevation=elang, azimuth=azang)

    # Adjust camera
    # f = mlab.gcf()
    # camera = f.scene.camera
    # camera.yaw(10)
    # camera.pitch(0)
    # camera.roll(10)
    # grab size in x and y if sz is list
    if type(SZ) is int:
        SZX = SZ;
        SZY = SZ
    elif type(SZ) is list:
        SZX = SZ[0];
        SZY = SZ[1]
    else:
        try:
            SZX = SZ[0];
            SZY = SZ[1]
        except:
            SZX = SZ;
            SZY = SZ

    if fname == '':
        print 'fname =', fname, '\n --> so showing image...'
        mlab.show()
    else:
        print 'fname =', fname, '\n --> so saving image...'
        mlab.savefig(fname, size=(SZX, SZY))

    mlab.close('all')



def pf_display_2panel(x, y, C0, C1, title0, title1='', vmin='auto', vmax='auto', ptsz=5, cmap=cm.CMRmap, axis_on=True,
                      close=True):
    """Display then close a scatter plot of two scalar quantities C0, C1"""
    fig, ax = plt.subplots(1, 2)
    if isinstance(vmin, str):
        vmin = min(np.nanmin(C0[:]), np.nanmin(C1[:]))
        print 'vmin=', vmin
    if isinstance(vmax, str):
        vmax = max(np.nanmax(C0[:]), np.nanmax(C1[:]))
        print 'vmax=', vmax
    # scatter scale (for color scale)
    scsc0 = ax[0].scatter(x, y, c=C0, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    scsc1 = ax[1].scatter(x, y, c=C1, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[0].set_title(title0)
    ax[1].set_title(title1)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    if np.nanmax(C0) > np.nanmax(C1):
        print 'maxC0>maxC1'
        fig.colorbar(scsc0, cax=cbar_ax)
    else:
        fig.colorbar(scsc1, cax=cbar_ax)
    if not axis_on:
        ax[0].axis('off')
        ax[1].axis('off')
    plt.show()
    if close:
        plt.close('all')


def pf_display_vector(x, y, C0, C1, varchar, title='', subscripts='cartesian', vmin='auto', vmax='auto', ptsz=5,
                      cmap=cm.CMRmap, axis_on=True, close=True):
    """Display then close a scatter plot of components of vector quantity C

    Parameters
    ----------
    x,y : Nx1 arrays
        positions of evaluated points
    C0,C1 : Nx1 arrays
        components of evalauated vector
    varchar : string
        the name of the tensorial variable (raw string works for LaTeX)
    title : string
        additional title above all subplots
    subscripts : string ('cartesian','polar')
        puts subscripts on the subtitles (ie '_x', etc).
        If 'theory', then compares
    """
    fig, ax = plt.subplots(1, 2)
    if isinstance(vmin, str):
        vmin = min(np.nanmin(C0[:]), np.nanmin(C1[:]))
        print 'vmin=', vmin
    if isinstance(vmax, str):
        vmax = max(np.nanmax(C0[:]), np.nanmax(C1[:]))
        print 'vmax=', vmax
    # scatter scale (for color scale)
    scsc0 = ax[0].scatter(x, y, c=C0, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    scsc1 = ax[1].scatter(x, y, c=C1, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    if np.nanmax(C0) > np.nanmax(C1):
        print 'maxC0>maxC1'
        fig.colorbar(scsc0, cax=cbar_ax)
    else:
        fig.colorbar(scsc1, cax=cbar_ax)
    if subscripts == 'cartesian':
        ax[0].set_title(r'${}_x$'.format(varchar))
        ax[1].set_title(r'${}_y$'.format(varchar))
    elif subscripts == 'polar':
        ax[0].set_title(r'${}_r$'.format(varchar))
        ax[1].set_title(r'${}_\theta$'.format(varchar))
    fig.text(0.5, 0.975, title, horizontalalignment='center', verticalalignment='top')
    if not axis_on:
        ax[0].axis('off')
        ax[1].axis('off')

    plt.show()
    if close:
        plt.close('all')


def pf_display_4panel(x, y, C0, C1, C2, C3, title0, title1='', title2='', title3='', vmin='auto', vmax='auto', ptsz=5,
                      cmap=cm.CMRmap, axis_on=True, close=True):
    """Display then close a scatter plot of four scalar quantities C0, C1, C2, C3"""
    fig, ax = plt.subplots(2, 2)
    if isinstance(vmin, str):
        vmin = min(np.nanmin(C0[:]), np.nanmin(C1[:]))
        print 'vmin=', vmin
    if isinstance(vmax, str):
        vmax = max(np.nanmax(C0[:]), np.nanmax(C1[:]))
        print 'vmax=', vmax
    # scatter scale (for color scale)
    scsc0 = ax[0, 0].scatter(x, y, c=C0, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    ax[0, 1].scatter(x, y, c=C1, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    ax[1, 0].scatter(x, y, c=C2, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    ax[1, 1].scatter(x, y, c=C3, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    ax[0, 0].set_aspect('equal');
    ax[1, 0].set_aspect('equal')
    ax[1, 0].set_aspect('equal');
    ax[0, 1].set_aspect('equal')
    ax[0, 0].set_title(title0);
    ax[0, 1].set_title(title1)
    ax[1, 0].set_title(title2);
    ax[1, 1].set_title(title3)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(scsc0, cax=cbar_ax)
    if not axis_on:
        ax[0, 0].axis('off')
        ax[0, 1].axis('off')
        ax[1, 0].axis('off')
        ax[1, 1].axis('off')
    plt.show()
    if close:
        plt.close('all')


def pf_display_tensor(x, y, C0, C1, C2, C3, varchar, title='', subscripts='cartesian', vmin='auto', vmax='auto', ptsz=5,
                      cmap=cm.CMRmap, axis_on=True, close=True):
    """Display then close a scatter plot of the 2x2 tensor C with components C0,C1,C2,C3.

    Parameters
    ----------
    x,y : Nx1 arrays
        positions of evaluated points
    C0,C1,C2,C3 : Nx1 arrays
        components of evalauated tensor
    varchar : string
        the name of the tensorial variable (raw string works for LaTeX)
    subscripts : string ('cartesian','polar','cartesiantensortheory','cartesianvectortheory','polartensortheory','polarvectortheory')
        puts subscripts on the subtitles (ie '_xx', etc)
    title : string
        additional title above all subplots
    """
    fig, ax = plt.subplots(2, 2)
    if isinstance(vmin, str):
        vmin = min(np.nanmin(C0[:]), np.nanmin(C1[:]))
        print 'vmin=', vmin
    if isinstance(vmax, str):
        vmax = max(np.nanmax(C0[:]), np.nanmax(C1[:]))
        print 'vmax=', vmax
    # scatter scale (for color scale)
    scsc0 = ax[0, 0].scatter(x, y, c=C0, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    ax[0, 1].scatter(x, y, c=C1, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    ax[1, 0].scatter(x, y, c=C2, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    ax[1, 1].scatter(x, y, c=C3, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    ax[0, 0].set_aspect('equal');
    ax[1, 0].set_aspect('equal')
    ax[1, 0].set_aspect('equal');
    ax[0, 1].set_aspect('equal')

    if subscripts == 'cartesian':
        ax[0, 0].set_title(r'${}$'.format(varchar) + r'$_{xx}$')
        ax[0, 1].set_title(r'${}$'.format(varchar) + r'$_{xy}$')
        ax[1, 0].set_title(r'${}$'.format(varchar) + r'$_{yx}$')
        ax[1, 1].set_title(r'${}$'.format(varchar) + r'$_{yy}$')
    elif subscripts == 'polar':
        ax[0, 0].set_title(r'${}$'.format(varchar) + r'$_{r r}$')
        ax[0, 1].set_title(r'${}$'.format(varchar) + r'$_{r \theta}$')
        ax[1, 0].set_title(r'${}$'.format(varchar) + r'$_{\theta r}$')
        ax[1, 1].set_title(r'${}$'.format(varchar) + r'$_{\theta\theta}$')
    elif subscripts == 'cartesiantensortheory':
        ax[0, 0].set_title(r'${}$'.format(varchar) + r'$_{xx}$')
        ax[0, 1].set_title(r'${}$'.format(varchar) + r'$_{yy}$')
        ax[1, 0].set_title(r'${}$'.format(varchar) + r'$_{xx}$ theory')
        ax[1, 1].set_title(r'${}$'.format(varchar) + r'$_{yy}$ theory')
    elif subscripts == 'cartesianvectortheory':
        ax[0, 0].set_title(r'${}$'.format(varchar) + r'$_{x}$')
        ax[0, 1].set_title(r'${}$'.format(varchar) + r'$_{y}$')
        ax[1, 0].set_title(r'${}$'.format(varchar) + r'$_{x}$ theory')
        ax[1, 1].set_title(r'${}$'.format(varchar) + r'$_{y}$ theory')
    elif subscripts == 'polarvectortheory':
        ax[0, 0].set_title(r'${}$'.format(varchar) + r'$_{r}$')
        ax[0, 1].set_title(r'${}$'.format(varchar) + r'$_{\theta}$')
        ax[1, 0].set_title(r'${}$'.format(varchar) + r'$_{r}$ theory')
        ax[1, 1].set_title(r'${}$'.format(varchar) + r'$_{\theta}$ theory')
    elif subscripts == 'polartensortheory':
        ax[0, 0].set_title(r'${}$'.format(varchar) + r'$_{r r}$')
        ax[0, 1].set_title(r'${}$'.format(varchar) + r'$_{\theta\theta}$')
        ax[1, 0].set_title(r'${}$'.format(varchar) + r'$_{r r}$ theory')
        ax[1, 1].set_title(r'${}$'.format(varchar) + r'$_{\theta\theta}$ theory')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(scsc0, cax=cbar_ax)
    fig.text(0.5, 0.975, title, horizontalalignment='center', verticalalignment='top')
    if not axis_on:
        ax[0, 0].axis('off')
        ax[0, 1].axis('off')
        ax[1, 0].axis('off')
        ax[1, 1].axis('off')
    plt.show()
    if close:
        plt.close('all')


def pf_plot_pcolormesh_scalar(x, y, C, outdir, name, ind, title, title2='', subtext='', subsubtext='',
                              vmin='auto', vmax='auto', ptsz=10, cmap=cm.CMRmap, shape='circle'):
    """Save a single-panel plot of a scalar quantity C as colored pcolormesh

    Parameters
    ----------
    x, y : NxN mesh arrays
        the x and y positions of the points evaluated to Cx, Cy
    C : NxN arrays
        values for the plotted quantity C evaluated at points (x,y)
    outdir : string
        where to save the img
    name : string
        the name of the variable --> file will be saved as name_ind#.png
    varchar : string
        the variable name as a character (could be LaTeX formatted)
    ind : int
        index number for the image
    title : string
    title2 : string
        placed below title
    subtext : string
        placed below plot
    subsubtext : string
        placed at bottom of image
    vmin, vmax : float
        minimum, maximum value of C for colorbar; default is range of values in C
    ptsz : float
        size of colored marker (dot)
    """
    fig, ax = plt.subplots(1, 1)
    if isinstance(vmin, str):
        vmin = np.nanmin(C)
    if isinstance(vmax, str):
        vmax = np.nanmax(C)
    # scatter scale (for color scale)
    scsc = ax.pcolormesh(x, y, C, cmap=cmap, vmin=vmin, vmax=vmax)
    R = x.max()
    if shape == 'circle':
        t = np.arange(0, 2 * np.pi + 0.01, 0.01)
        plt.plot(R * np.cos(t), R * np.sin(t), 'k-')
    elif shape == 'square':
        t = np.array([-R, R])
        plt.plot(R * np.array([1, 1]), t, 'k-')
        plt.plot(t, R * np.array([1, 1]), 'k-')
        plt.plot(t, -R * np.array([1, 1]), 'k-')
        plt.plot(-R * np.array([1, 1]), t, 'k-')
    elif shape == 'unitsq':
        t = np.array([0, R])
        plt.plot(R * np.array([1, 1]), t, 'k-')
        plt.plot(t, R * np.array([0, 1]), 'k-')
        plt.plot(t, -R * np.array([0, 1]), 'k-')
        plt.plot(np.array([0, 0]), t, 'k-')
    elif shape == 'rectangle2x1':
        t = np.array([-R, R])
        plt.plot(R * np.array([1, 1]), 2 * t, 'k-')
        plt.plot(t, 2 * R * np.array([1, 1]), 'k-')
        plt.plot(t, -2 * R * np.array([1, 1]), 'k-')
        plt.plot(-R * np.array([1, 1]), 2 * t, 'k-')

    ax.set_aspect('equal')
    ax.axis('off');
    ax.set_title(title)
    fig.text(0.5, 0.12, subtext, horizontalalignment='center')
    fig.text(0.5, 0.05, subsubtext, horizontalalignment='center')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(scsc, cax=cbar_ax)
    fig.text(0.5, 0.98, title2, horizontalalignment='center', verticalalignment='top')
    savedir = prepdir(outdir)
    plt.savefig(savedir + name + '_' + '{0:06d}'.format(ind) + '.png')
    plt.close('all')


def pf_plot_scatter_scalar(x, y, C, outdir, name, ind, title, title2='', subtext='', subsubtext='', vmin='auto',
                           vmax='auto', ptsz=10, cmap=cm.CMRmap, shape='none', ticks='off'):
    """Save a single-panel plot of a scalar quantity C as colored scatterplot

    Parameters
    ----------
    x, y : Nx1 arrays
        the x and y positions of the points evaluated to Cx, Cy
    C : Nx1 arrays
        values for the plotted quantity C evaluated at points (x,y)
    outdir : string
        where to save the img.
    name : string
        the name of the variable --> file will be saved as name_ind#.png
    varchar : string
        the variable name as a character (could be LaTeX formatted)
    ind : int
        index number for the image
    title : string
        string title placed above image
    title2 : string
        placed below title
    subtext : string
        placed below plot
    subsubtext : string
        placed at bottom of image
    vmin, vmax : float
        minimum, maximum value of C for colorbar; default is range of values in C
    ptsz : float
        size of colored marker (dot)
    shape : string ('circle', 'square', 'unitsq', etc)
        characterization of the border to draw, default is 'none' --> no border
    ticks : string ('on' or 'off')
        whether or not to plot the axis (and tick marks)
    """
    fig, ax = plt.subplots(1, 1)
    if isinstance(vmin, str):
        vmin = np.nanmin(C)

    if isinstance(vmax, str):
        vmax = np.nanmax(C)

    # scatter scale (for color scale)
    scsc = ax.scatter(x, y, c=C, s=ptsz, cmap=cmap, edgecolor='', vmin=vmin, vmax=vmax)
    # scsc = ax.pcolormesh(x, y, C, cmap=cmap, vmin=vmin, vmax=vmax)
    R = x.max()
    if shape == 'circle':
        t = np.arange(0, 2 * np.pi + 0.01, 0.01)
        plt.plot(R * np.cos(t), R * np.sin(t), 'k-')
    elif shape == 'square':
        t = np.array([-R, R])
        plt.plot(R * np.array([1, 1]), t, 'k-')
        plt.plot(t, R * np.array([1, 1]), 'k-')
        plt.plot(t, -R * np.array([1, 1]), 'k-')
        plt.plot(-R * np.array([1, 1]), t, 'k-')
    elif shape == 'unitsq':
        t = np.array([0, R])
        plt.plot(R * np.array([1, 1]), t, 'k-')
        plt.plot(t, R * np.array([1, 1]), 'k-')
        plt.plot(t, np.array([0, 0]), 'k-')
        plt.plot(np.array([0, 0]), t, 'k-')
    elif shape == 'rectangle2x1':
        t = np.array([-R, R])
        plt.plot(R * np.array([1, 1]), 2 * t, 'k-')
        plt.plot(t, 2 * R * np.array([1, 1]), 'k-')
        plt.plot(t, -2 * R * np.array([1, 1]), 'k-')
        plt.plot(-R * np.array([1, 1]), 2 * t, 'k-')
    elif shape == 'rectangle1x2':
        t = np.array([-R, R])
        plt.plot(2 * R * np.array([1, 1]), t, 'k-')
        plt.plot(2 * t, R * np.array([1, 1]), 'k-')
        plt.plot(2 * t, -R * np.array([1, 1]), 'k-')
        plt.plot(-2 * R * np.array([1, 1]), t, 'k-')

    ax.set_aspect('equal')
    if ticks == 'off':
        ax.axis('off')
    ax.set_title(title)
    fig.text(0.5, 0.12, subtext, horizontalalignment='center')
    fig.text(0.5, 0.05, subsubtext, horizontalalignment='center')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(scsc, cax=cbar_ax)
    fig.text(0.5, 0.98, title2, horizontalalignment='center', verticalalignment='top')
    savedir = prepdir(outdir)
    plt.savefig(savedir + name + '_' + '{0:06d}'.format(ind) + '.png')
    plt.close('all')


def pf_plot_scatter_2panel(x, y, C0, C1, outdir, name, varchar, ind, title, title2='', subtext='', subsubtext='',
                           vmin='auto', vmax='auto', ptsz=10, subscripts='cartesian', shape='', cmap=cm.CMRmap):
    """Save a two-panel plot of the components of a vector quantity (C0,C1) as colored scatterplot

    Parameters
    ----------
    x, y : Nx1 arrays
        the x and y positions of the points evaluated to C0, C1
    C0, C1 : Nx1 arrays
        values for the two plotted quantities evaluated at points (x,y)
    outdir : string
        where to save the img
    name : string
        the name of the variable --> file will be saved as name_ind#.png
    varchar : string
        the variable name as a character (could be LaTeX formatted)
    ind : int
        index number for the image
    title : string
    title2 : string
        placed below title
    subtext : string
        placed below plot
    subsubtext : string
        placed at bottom of image
    vmin : float
        minimum value of Cx or Cy for colorbar. Default is string 'auto', which prompts function to take min of C0
    ptsz : float
        size of colored marker (dot)
    subscripts : str ('cartesian', 'polar', 'theory')
        what coordinate system (and/or subset of elements) to use for naming the subplots; default is 'cartesian' (x,y), can be 'polar' (r,\theta)
    """
    # Plot and save u
    fig, ax = plt.subplots(1, 2)
    if isinstance(vmin, str):
        vmin = np.min([np.nanmin(C0), np.nanmin(C1)])
    if isinstance(vmax, str):
        vmax = np.max([np.nanmax(C0), np.nanmax(C1)])
    scu = ax[0].scatter(x, y, c=C0, s=ptsz, edgecolor='', vmin=vmin, vmax=vmax, cmap=cmap)
    ax[1].scatter(x, y, c=C1, s=ptsz, edgecolor='', vmin=vmin, vmax=vmax, cmap=cmap)
    R = x.max()
    if shape == 'circle':
        t = np.arange(0, 2 * np.pi + 0.01, 0.01)
        ax[0].plot(R * np.cos(t), R * np.sin(t), 'k-')
        ax[1].plot(R * np.cos(t), R * np.sin(t), 'k-')
    elif shape == 'unitsq':
        t = np.array([0, R])
        ax[0].plot(R * np.array([1, 1]), t, 'k-')
        ax[0].plot(t, R * np.array([1, 1]), 'k-')
        ax[0].plot(t, np.array([0, 0]), 'k-')
        ax[0].plot(np.array([0, 0]), t, 'k-')
        ax[1].plot(R * np.array([1, 1]), t, 'k-')
        ax[1].plot(t, R * np.array([1, 1]), 'k-')
        ax[1].plot(t, np.array([0, 0]), 'k-')
        ax[1].plot(np.array([0, 0]), t, 'k-')
    elif shape == 'square':
        t = np.array([-R, R])
        ax[0].plot(R * np.array([1, 1]), t, 'k-')
        ax[0].plot(t, R * np.array([1, 1]), 'k-')
        ax[0].plot(t, -R * np.array([1, 1]), 'k-')
        ax[0].plot(-R * np.array([1, 1]), t, 'k-')
        ax[1].plot(R * np.array([1, 1]), t, 'k-')
        ax[1].plot(t, R * np.array([1, 1]), 'k-')
        ax[1].plot(t, -R * np.array([1, 1]), 'k-')
        ax[1].plot(-R * np.array([1, 1]), t, 'k-')
    elif shape == 'rectangle2x1':
        t = np.array([-R, R])
        ax[0].plot(R * np.array([1, 1]), 2 * t, 'k-')
        ax[0].plot(t, 2 * R * np.array([1, 1]), 'k-')
        ax[0].plot(t, -2 * R * np.array([1, 1]), 'k-')
        ax[0].plot(-R * np.array([1, 1]), 2 * t, 'k-')
        ax[1].plot(R * np.array([1, 1]), 2 * t, 'k-')
        ax[1].plot(t, 2 * R * np.array([1, 1]), 'k-')
        ax[1].plot(t, -2 * R * np.array([1, 1]), 'k-')
        ax[1].plot(-R * np.array([1, 1]), 2 * t, 'k-')
    elif shape == 'rectangle2x1':
        t = np.array([-R, R])
        ax[0].plot(2 * R * np.array([1, 1]), t, 'k-')
        ax[0].plot(2 * t, R * np.array([1, 1]), 'k-')
        ax[0].plot(2 * t, -R * np.array([1, 1]), 'k-')
        ax[0].plot(-2 * R * np.array([1, 1]), t, 'k-')
        ax[1].plot(2 * R * np.array([1, 1]), t, 'k-')
        ax[1].plot(2 * t, R * np.array([1, 1]), 'k-')
        ax[1].plot(2 * t, -R * np.array([1, 1]), 'k-')
        ax[1].plot(-2 * R * np.array([1, 1]), t, 'k-')

    ax[0].set_aspect('equal');
    ax[1].set_aspect('equal');
    ax[0].axis('off');
    ax[1].axis('off');
    if subscripts == 'cartesian':
        ax[0].set_title(r'${}_x$'.format(varchar))
        ax[1].set_title(r'${}_y$'.format(varchar))
    elif subscripts == 'polar':
        ax[0].set_title(r'${}_r$'.format(varchar))
        ax[1].set_title(r'${}_{\theta}$'.format(varchar))
    elif subscripts == 'theory':
        ax[0].set_title(r'${}$ exprmt'.format(varchar))
        ax[1].set_title(r'${}$ theory'.format(varchar))

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(scu, cax=cbar_ax)
    fig.text(0.5, 0.9, title2, horizontalalignment='center', verticalalignment='top')
    fig.text(0.5, 0.05, subsubtext, horizontalalignment='center')
    fig.text(0.5, 0.15, subtext, horizontalalignment='center')
    fig.text(0.5, 0.975, title, horizontalalignment='center', verticalalignment='top')
    savedir = prepdir(outdir)
    plt.savefig(savedir + name + '_' + '{0:06d}'.format(ind) + '.png')
    plt.close('all')


def pf_plot_scatter_4panel(x, y, C00, C01, C10, C11, outdir, name, varchar, ind, title, title2, subtext, subsubtext,
                           vmin='auto', vmax='auto', ptsz=3, subscripts='cartesian', cmap=cm.CMRmap, shape='none'):
    """Plot and save a four-panel plot of the components of a tensor quantity as colored scatterplot

    Parameters
    ----------
    x, y : Nx1 arrays
        the x and y positions of the points evaluated to C
    C00, C01, C10, C11 : Nx1 arrays
        values for the plotted vector evaluated at points (x,y)
    outdir : string
        where to save the img
    name : string
        the name of the variable --> file will be saved as name_ind#.png
    varchar : string
        the variable name as a character (could be LaTeX formatted)
    ind : int
        index number for the image
    title : string
    title2 : string
        placed below title
    subtext : string
        placed below plot
    subsubtext : string
        placed at bottom of image
    vmin : float
        minimum value of Cij for colorbar. Default is string 'auto', which prompts function to take min of C00
    ptsz : float
        size of colored marker (dot)
    subscripts : str ('cartesian', 'polar', 'cartesiantheory', 'polartheory', 'cartesianvectortheory')
        what coordinate system (and/or subset of elements) to use for naming the subplots; default is 'cartesian' (x,y), can be 'polar' (r,\theta)
    """
    fig, ax = plt.subplots(2, 2)
    if isinstance(vmin, str):
        vmin = np.min([np.nanmin(C00), np.nanmin(C01), np.nanmin(C10), np.nanmin(C11)])
    if isinstance(vmax, str):
        vmax = np.max([np.nanmax(C00), np.nanmax(C01), np.nanmax(C10), np.nanmax(C11)])
    ptsz = 10
    sccarte = ax[0, 0].scatter(x, y, c=C00, s=ptsz, edgecolor='', vmin=vmin, vmax=vmax, cmap=cmap)
    ax[0, 1].scatter(x, y, c=C01, s=ptsz, edgecolor='', vmin=vmin, vmax=vmax, cmap=cmap)
    ax[1, 0].scatter(x, y, c=C10, s=ptsz, edgecolor='', vmin=vmin, vmax=vmax, cmap=cmap)
    ax[1, 1].scatter(x, y, c=C11, s=ptsz, edgecolor='', vmin=vmin, vmax=vmax, cmap=cmap)
    R = x.max()
    if shape == 'circle':
        t = np.arange(0, 2 * np.pi + 0.01, 0.01)
        ax[0, 0].plot(R * np.cos(t), R * np.sin(t), 'k-')
        ax[0, 1].plot(R * np.cos(t), R * np.sin(t), 'k-')
        ax[1, 0].plot(R * np.cos(t), R * np.sin(t), 'k-')
        ax[1, 1].plot(R * np.cos(t), R * np.sin(t), 'k-')
    elif shape == 'unitsq':
        t = np.array([0, R])
        ax[0, 0].plot(R * np.array([1, 1]), t, 'k-')
        ax[0, 0].plot(t, R * np.array([1, 1]), 'k-')
        ax[0, 0].plot(t, np.array([0, 0]), 'k-')
        ax[0, 0].plot(np.array([0, 0]), t, 'k-')
        ax[1, 0].plot(R * np.array([1, 1]), t, 'k-')
        ax[1, 0].plot(t, R * np.array([1, 1]), 'k-')
        ax[1, 0].plot(t, np.array([0, 0]), 'k-')
        ax[1, 0].plot(np.array([0, 0]), t, 'k-')
        ax[0, 1].plot(R * np.array([1, 1]), t, 'k-')
        ax[0, 1].plot(t, R * np.array([1, 1]), 'k-')
        ax[0, 1].plot(t, np.array([0, 0]), 'k-')
        ax[0, 1].plot(np.array([0, 0]), t, 'k-')
        ax[1, 1].plot(R * np.array([1, 1]), t, 'k-')
        ax[1, 1].plot(t, R * np.array([1, 1]), 'k-')
        ax[1, 1].plot(t, np.array([0, 0]), 'k-')
        ax[1, 1].plot(np.array([0, 0]), t, 'k-')
    elif shape == 'square':
        sfY = 1.0
        sfX = 1.0
    elif shape == 'rectangle2x1':
        sfX = 1.0
        sfY = 2.0
    elif shape == 'rectangle1x2':
        sfX = 2.0
        sfY = 1.0

    if shape == 'square' or shape == 'rectangle2x1' or shape == 'rectangle1x2':
        t = np.array([-R, R])
        s = R * np.array([1, 1])
        ax[0, 0].plot(sfX * s, sfY * t, 'k-')
        ax[0, 0].plot(sfX * t, sfY * s, 'k-')
        ax[0, 0].plot(sfX * t, -sfY * s, 'k-')
        ax[0, 0].plot(-sfX * s, sfY * t, 'k-')
        ax[0, 1].plot(sfX * s, sfY * t, 'k-')
        ax[0, 1].plot(sfX * t, sfY * s, 'k-')
        ax[0, 1].plot(sfX * t, -sfY * s, 'k-')
        ax[0, 1].plot(-sfX * s, sfY * t, 'k-')
        ax[1, 0].plot(sfX * s, sfY * t, 'k-')
        ax[1, 0].plot(sfX * t, sfY * s, 'k-')
        ax[1, 0].plot(sfX * t, -sfY * s, 'k-')
        ax[1, 0].plot(-sfX * s, sfY * t, 'k-')
        ax[1, 1].plot(sfX * s, sfY * t, 'k-')
        ax[1, 1].plot(sfX * t, sfY * s, 'k-')
        ax[1, 1].plot(sfX * t, -sfY * s, 'k-')
        ax[1, 1].plot(-sfX * s, sfY * t, 'k-')

    if subscripts == 'cartesian':
        ax[0, 0].set_title(r'${}$'.format(varchar) + r'$_{xx}$')
        ax[0, 1].set_title(r'${}$'.format(varchar) + r'$_{xy}$')
        ax[1, 0].set_title(r'${}$'.format(varchar) + r'$_{yx}$')
        ax[1, 1].set_title(r'${}$'.format(varchar) + r'$_{yy}$')
    elif subscripts == 'polar':
        ax[0, 0].set_title(r'${}$'.format(varchar) + r'$_{r r}$')
        ax[0, 1].set_title(r'${}$'.format(varchar) + r'$_{r \theta}$')
        ax[1, 0].set_title(r'${}$'.format(varchar) + r'$_{\theta r}$')
        ax[1, 1].set_title(r'${}$'.format(varchar) + r'$_{\theta\theta}$')
    elif subscripts == 'cartesiantensortheory':
        ax[0, 0].set_title(r'${}$'.format(varchar) + r'$_{xx}$')
        ax[0, 1].set_title(r'${}$'.format(varchar) + r'$_{yy}$')
        ax[1, 0].set_title(r'${}$'.format(varchar) + r'$_{xx}$ theory')
        ax[1, 1].set_title(r'${}$'.format(varchar) + r'$_{yy}$ theory')
    elif subscripts == 'cartesianvectortheory':
        ax[0, 0].set_title(r'${}$'.format(varchar) + r'$_{x}$')
        ax[0, 1].set_title(r'${}$'.format(varchar) + r'$_{y}$')
        ax[1, 0].set_title(r'${}$'.format(varchar) + r'$_{x}$ theory')
        ax[1, 1].set_title(r'${}$'.format(varchar) + r'$_{y}$ theory')
    elif subscripts == 'polarvectortheory':
        ax[0, 0].set_title(r'${}$'.format(varchar) + r'$_{r}$')
        ax[0, 1].set_title(r'${}$'.format(varchar) + r'$_{\theta}$')
        ax[1, 0].set_title(r'${}$'.format(varchar) + r'$_{r}$ theory')
        ax[1, 1].set_title(r'${}$'.format(varchar) + r'$_{\theta}$ theory')
    elif subscripts == 'polartensortheory':
        ax[0, 0].set_title(r'${}$'.format(varchar) + r'$_{r r}$')
        ax[0, 1].set_title(r'${}$'.format(varchar) + r'$_{\theta\theta}$')
        ax[1, 0].set_title(r'${}$'.format(varchar) + r'$_{r r}$ theory')
        ax[1, 1].set_title(r'${}$'.format(varchar) + r'$_{\theta\theta}$ theory')

    ax[0, 0].axis('off');
    ax[0, 1].axis('off');
    ax[1, 0].axis('off');
    ax[1, 1].axis('off')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(sccarte, cax=cbar_ax)
    # global title
    fig.text(0.5, 0.9, title2, horizontalalignment='center', verticalalignment='top')
    fig.text(0.5, 0.05, subsubtext, horizontalalignment='center')
    fig.text(0.5, 0.15, subtext, horizontalalignment='center')
    fig.text(0.5, 0.975, title, horizontalalignment='center', verticalalignment='top')
    savedir = prepdir(outdir)
    plt.savefig(savedir + name + '_' + '{0:06d}'.format(ind) + '.png')
    plt.close('all')


def pf_plot_1D(x, functs, outdir, name, xlab, ylab, title, labels, ptsz=5):
    """Plot and save a 1D plot with any number of curves (columns of numpy array functs).

    Parameters
    ----------
    x : Nx1 array
        the x-axis values
    functs : NxM array
        Each column is a set of values to plot against x
    outdir : string
        where to save the img
    name : string
        the name of the output image --> file will be saved as name.png
    xlab, ylab : strings
        the x and y labels
    title : string
    title2 : string
        placed below title
    subtext : string
        placed below plot
    subsubtext : string
        placed at bottom of image
    labels : dictionary
        The labels for each column of functs, with keys 0,1,2,3...
    ptsz : float
        size of colored marker (dot)
    """
    # style.use('ggplot')
    fig, ax = plt.subplots(1, 1)
    # Colors: red (could use #DD6331 like in publication), green, purple, yellow, blue, orange
    ax.set_color_cycle(['#B32525', '#77AC30', '#7E2F8E', '#EFBD46', '#0E7ABF', '#D95419'])
    ind = 0
    for funct in functs.T:
        # print funct
        plt.plot(x, funct, '.', label=labels[ind])
        ind += 1
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    savedir = prepdir(outdir)
    legend = ax.legend(loc='best')
    plt.savefig(savedir + name + '_' + '{0:06d}'.format(ind) + '.png')
    plt.close('all')


def pf_contour(x, y, z, level, Bxy, outdir, name, ind, title, title2, subtext, subsubtext):
    """Plot contour of z on at value=level.
    """
    fig, ax = plt.subplots(1, 1)
    fig.text(0.5, 0.96, title2, horizontalalignment='center', verticalalignment='top')
    fig.text(0.5, 0.02, subsubtext, horizontalalignment='center')
    fig.text(0.5, 0.05, subtext, horizontalalignment='center')
    fig.text(0.5, 0.99, title, horizontalalignment='center', verticalalignment='top')
    ax.contour(x, y, z, levels=[level], colors='k')
    ax.plot(Bxy[:, 0], Bxy[:, 1], 'k-')
    ax.axis('equal')
    ax.axis('off')
    return fig, ax


def pf_contour_unstructured(x, y, z, n, level, Bxy, outdir, name, ind, title, title2, subtext, subsubtext):
    """Interpolate data (x,y,z) onto uniform grid of dim nxn, then plot as contour plot at value=level"""
    X, Y, Z = pfh.interpol_meshgrid(x, y, z, n)
    pf_contour(X, Y, Z, level, Bxy, outdir, name, ind, title, title2, subtext, subsubtext)


def pf_add_contour_to_plot(x, y, z, n, level, ax, color='k'):
    """Interpolate data (x,y,z) onto uniform grid of dim nxn, then add contour to axis ax (plotting contour of value=level)"""
    X, Y, Z = pfh.interpol_meshgrid(x, y, z, n)
    ax.contour(X, Y, Z, levels=[level], colors=color)


def pf_add_contourf_to_plot(x, y, z, n, level, ax, color='k'):
    """Interpolate data (x,y,z) onto uniform grid of dim nxn, then add contour to axis ax (plotting contour of value=level)"""
    X, Y, Z = pfh.interpol_meshgrid(x, y, z, n)
    ax.contourf(X, Y, Z, levels=[0, level], colors=color)
