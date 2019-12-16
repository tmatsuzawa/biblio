import numpy as np
import lepm.plotting.plotting as leplt
import lepm.plotting.colormaps as lecmaps
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import lepm.data_handling as dh
import lepm.stringformat as sf
import pickle as pkl
import glob
import lepm.dataio as dio
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
import argparse

parser = argparse.ArgumentParser(description='Solve for poential flow around a shell in 2d.')
parser.add_argument('-vtb', '--vtb', help='Velocity at top and bottom of the sample', type=float, default=1.0)
parser.add_argument('-geometry', '--geometry', help='Shell or dolphin, etc', type=str, default='shell')
parser.add_argument('-dshell', '--dshell', help='Size of the shell', type=float, default=0.5)
parser.add_argument('-thole', '--theta_hole', help='Half size of the hole in radians', type=float, default=0.5)
parser.add_argument('-phi', '--phi', help='Angle at which center of hole points, in pi radians', type=float, default=0.)
parser.add_argument('-N', '--N', help='Number of points in the mesh (rough)', type=int, default=1000)
parser.add_argument('-R', '--R', help='Radius (or half-width) of the mesh', type=float, default=1.0)
parser.add_argument('-thick', '--thickness', help='Thickness of the shell', type=float, default=.10)
args = parser.parse_args()


def gray_out_polygon(ax, polygon, color='gray', zorder=1000):
    # Grey out the specified region
    # Create a bunch of vertical lines and find intersections with linesegs of polygon
    # xxs = np.linspace(np.min(polygon[:, 0]), np.max(polygon[:, 1]), npts, endpoint=True)
    # vsegs = np.dstack((xxs, np.zeros_like(xxs), xxs, np.ones_like(xxs)))
    # psegs = lsegs.array_to_linesegs(polygon)
    # intx = lsegs.linesegs_intersect_linesegs(vsegs, psegs)
    # for key in intx:
    #     # for each key, sort the intersections by y value and fill between
    #     intx[key][1] --> actually this is hard...
    # ax.fill_between(xfill, yfill[0], yfill[1], color=color, alpha=1.0, zorder=zorder, lw=0)
    patch = Polygon(polygon, closed=True, alpha=1.0, color=color, zorder=zorder)
    ax.add_patch(patch)


# Parameters for the boundary condition
# determine how much of the compensating flux is coming from the top versus the bottom (in range 0 - 1).
# A value for vtopvbot of 0 means all bottom, 1 means all top
graycolor = lecmaps.light_gray()
big_eps = 0.02
thole = 0.5
vinfty = args.vtb
dshell = args.dshell
thick = args.thickness
Nmesh = args.N
thole = args.theta_hole

# Derived geometric variables
vstr = '{0:0.2f}'.format(vinfty).replace('.', 'p')

# Create mesh and define function space
# mesh = UnitSquareMesh(100, 100)
if args.geometry == 'shell':
    specstr = 'square_vNbcs_vinfty' + vstr + '_Nmesh' + str(Nmesh)
    meshspec = 'dshell' + sf.float2pstr(args.dshell) + '_thetahole' + sf.float2pstr(thole) + \
               '_phihole' + sf.float2pstr(args.phi) + '_thick' + sf.float2pstr(args.thickness)
    meshfile = './meshes/shell_N' + str(Nmesh) + '_n*_R1p000_' + meshspec + '.xml'
    print('searching for ', meshfile)
    meshfile = glob.glob(meshfile)[0]
    # Load the information about the linesegments that defined the boundary, rmv_lsegs
    meshparamfn = meshfile[0:-3] + 'pkl'
    with open(meshparamfn, 'rb') as fn:
        pdict = pkl.load(fn)

    rmv_lsegs = pdict['rmv_lsegs']
    polygon = pdict['polygon']
    gray_polygon = True
elif args.geometry == 'dolphin':
    specstr = 'square_dolphin_vNbcs_vinfty' + vstr + '_Nmesh' + str(Nmesh)
    # mesh = Mesh("./meshes/dolfin_fine.xml.gz")
    meshspec = 'dolphin'
    rmv_lsegs = []
    gray_polygon = False

##############################################
# Load Phi data
#############################################
outdir = './results/' + 'potential_flow_shell_' + specstr + '/' + meshspec + '/'
with open(outdir + 'data.pkl', "rb") as fn:
    res = pkl.load(fn)

xy = res['xy']
phiv = res['phi']
uuv = res['uu']
if args.geometry == 'dolphin':
    boundary = res['boundaries']
    # Split the boundary into inner and outer
    b_inner0 = np.where(np.logical_and(boundary[:, 0] > big_eps, boundary[:, 0] < 1 - big_eps))[0]
    b_inner1 = np.where(np.logical_and(boundary[:, 1] > big_eps, boundary[:, 1] < 1 - big_eps))[0]
    b_inner = np.intersect1d(b_inner0, b_inner1)
    b_outer = np.setdiff1d(np.arange(len(boundary), dtype=int), b_inner)
    # Order the outer boundary by polar angle
    btheta = np.arctan2(boundary[b_outer, 1], boundary[b_outer, 0])
    bouter_inds = np.argsort(btheta)
    bo_xy = boundary[b_outer, :]
    # Order the inside boundary by using lattice methods
    bi_xy = boundary[b_inner]

maxx = np.max(xy[:, 0])
maxy = np.max(xy[:, 1])
eps = 1e-2

##############################################
# Save image of Phi
#############################################
print(('xy = ', xy))
print(('phiv = ', phiv))
wfig, hfig = 90, 60
fig, ax, cax = leplt.initialize_1panel_cbar_cent(wfig, hfig, wsfrac=0.5, cbar_pos='right', wcbarfrac=0.05, hcbarfrac=0.6)
ax = leplt.plot_pcolormesh(xy[:, 0], xy[:, 1], phiv, 100, ax=ax, cax=cax, zorder=0)
ax.set_title(r'Potential flow, $\Phi$')
ax.yaxis.set_label_position("right")
cax.set_ylabel(r'$\Phi$', rotation=0)

# Grey out the shell
if gray_polygon:
    gray_out_polygon(ax, polygon)
else:
    ax.plot(bi_xy[:, 0], bi_xy[:, 1], 'k.')

ax.set_xlim(0, 1.)
ax.set_ylim(0, 1.)
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
plt.savefig(outdir + 'potential_flow_shell' + specstr + '_phi.png', dpi=300)

#############################################
# Save vector flow field as png
#############################################
ngrid = 15
xgrid, ygrid, z0grid = dh.interpol_meshgrid(xy[:, 0], xy[:, 1], uuv[:, 0], ngrid, method='nearest')
xgrid, ygrid, z1grid = dh.interpol_meshgrid(xy[:, 0], xy[:, 1], uuv[:, 1], ngrid, method='nearest')

# Plot using numpy array
# qq = ax.quiver(xy[::modn, 0], xy[::modn, 1], uuv_quiv[::modn, 0], uuv_quiv[::modn, 1], pivot='mid', units='inches')
z0quiv = z0grid * 100. / ngrid
z1quiv = z1grid * 100. / ngrid
qq = ax.quiver(xgrid, ygrid, z0quiv, z1quiv, pivot='mid', units='inches', zorder=1)
ax.set_xlim(0, 1.)
ax.set_ylim(0, 1.)
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
plt.savefig(outdir + 'potential_flow_shell' + specstr + '_flowfield.png', dpi=300)

#############################################
# Save vector field components as png
#############################################
print(('uuv = ', uuv))
print(('xy = ', np.shape(xy)))
print(('uuv = ', np.shape(uuv)))
wfig, hfig = 180., 90.
y0cbar_frac = 0.3
wcbarfrac = 0.02
hcbar_fracw = (1. - 2. * y0cbar_frac) / wcbarfrac * (hfig / wfig)
fig, axes, cax = leplt.initialize_2panel_1cbar_centy(wfig, hfig, wsfrac=0.3, wssfrac=0.3,
                                                     wcbarfrac=wcbarfrac, hcbar_fracw=hcbar_fracw,
                                                     y0cbar_frac=y0cbar_frac, x0cbar_frac=0.8)
magv = np.max(np.abs(uuv.ravel()))
for ii in [0, 1]:
    ax = leplt.plot_pcolormesh(xy[:, 0], xy[:, 1], uuv[:, ii], 200, ax=axes[ii], cax=cax, cmap='rwb0',
                               vmin=-magv, vmax=magv)
    axes[ii].set_xlim(0, 1.)
    axes[ii].set_ylim(0, 1.)
    axes[ii].xaxis.set_ticks([])
    axes[ii].yaxis.set_ticks([])

# Grey out the shell
if gray_polygon:
    gray_out_polygon(axes[0], polygon)
    gray_out_polygon(axes[1], polygon)
else:
    axes[0].plot(bi_xy[:, 0], bi_xy[:, 1], 'k.')
    axes[1].plot(bi_xy[:, 0], bi_xy[:, 1], 'k.')

axes[0].set_title(r'$u = \partial_x \Phi$')
axes[1].set_title(r'$u = \partial_y \Phi$')
ax.yaxis.set_label_position("right")
cax.set_ylabel(r'$\partial_\mu\Phi$', rotation=0)
plt.savefig(outdir + 'potential_flow_shell' + specstr + '_flowuv.png', dpi=300)
plt.close('all')

#############################################
# Speed of flow field as pcolormesh
#############################################
wfig, hfig = 90, 60
speedv = np.sqrt(uuv[:, 0] ** 2 + uuv[:, 1] ** 2)
fig, ax, cax = leplt.initialize_1panel_cbar_cent(wfig, hfig, wsfrac=0.5, cbar_pos='right', wcbarfrac=0.05, hcbarfrac=0.6)
ax = leplt.plot_pcolormesh(xy[:, 0], xy[:, 1], speedv, 200, ax=ax, cax=cax, zorder=0)
ax.set_title(r'Flow speed, $|\nabla\Phi|$')
ax.yaxis.set_label_position("right")
cax.set_ylabel(r'$|\nabla\Phi|$', rotation=0, labelpad=10)

# Grey out the shell
if gray_polygon:
    gray_out_polygon(ax, polygon)
else:
    ax.plot(bi_xy[:, 0], bi_xy[:, 1], 'k.')

ax.set_xlim(0, 1.)
ax.set_ylim(0, 1.)
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
plt.savefig(outdir + 'potential_flow_shell' + specstr + '_speed.png', dpi=300)

# Repeat in log scale
fig, ax, cax = leplt.initialize_1panel_cbar_cent(wfig, hfig, wsfrac=0.5, cbar_pos='right', wcbarfrac=0.05, hcbarfrac=0.6)
ax = leplt.plot_pcolormesh(xy[:, 0], xy[:, 1], np.log10(speedv), 200, ax=ax, cax=cax, zorder=0)
ax.set_title(r'Flow speed, $|\nabla\Phi|$')
ax.yaxis.set_label_position("right")
cax.set_ylabel(r'$\log_{10}|\nabla\Phi|$', rotation=90, labelpad=10)

# Grey out the shell
if gray_polygon:
    gray_out_polygon(ax, polygon)
else:
    ax.plot(bi_xy[:, 0], bi_xy[:, 1], 'k.')

ax.set_xlim(0, 1.)
ax.set_ylim(0, 1.)
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
plt.savefig(outdir + 'potential_flow_shell' + specstr + '_speedlog.png', dpi=300)
plt.close('all')

#############################################
# Save stream plot as png
#############################################
wfig, hfig = 90, 60
fig, ax = leplt.initialize_1panel_centered_fig(wfig, hfig, wsfrac=0.5)
ngrid = 500
xgrid, ygrid, ugrid = dh.interpol_meshgrid(xy[:, 0], xy[:, 1], uuv[:, 0], ngrid, method='nearest')
xgrid, ygrid, vgrid = dh.interpol_meshgrid(xy[:, 0], xy[:, 1], uuv[:, 1], ngrid, method='nearest')
speed = np.sqrt(ugrid ** 2 + vgrid ** 2)
lw = 5 * speed / speed.max()
ax.streamplot(xgrid, ygrid, ugrid, vgrid, density=0.9, color='k', linewidth=lw)
for pair in rmv_lsegs:
    inds0, inds1 = [0, 2], [1, 3]
    plt.plot(pair[inds0], pair[inds1], 'r-')

ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Grey out the piston
if gray_polygon:
    gray_out_polygon(ax, polygon)
else:
    ax.plot(bi_xy[:, 0], bi_xy[:, 1], 'k.')

ax.set_title('Streamlines for potential flow')
plt.savefig(outdir + 'potential_flow_shell' + specstr + '_flowuvstreamscaled.png', dpi=300)

# Plot without scaling by velocity
ax.cla()
ax.streamplot(xgrid, ygrid, ugrid, vgrid, density=0.9, color='k', linewidth=1)
for pair in rmv_lsegs:
    inds0, inds1 = [0, 2], [1, 3]
    plt.plot(pair[inds0], pair[inds1], 'r-')

ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Grey out the piston
if gray_polygon:
    gray_out_polygon(ax, polygon)
else:
    ax.plot(bi_xy[:, 0], bi_xy[:, 1], 'k.')

ax.set_title('Streamlines for potential flow')
plt.savefig(outdir + 'potential_flow_shell' + specstr + '_flowuvstream.png', dpi=300)
plt.close('all')

#############################################
# Plot flow at top
#############################################
wfig, hfig = 90, 60
fig, ax = leplt.initialize_1panel_centered_fig(wfig, hfig, wsfrac=0.5)
top = np.where(xy[:, 1] > maxy - eps)[0]
bot = np.where(xy[:, 1] < eps)[0]
ax.scatter(xy[top, 0], uuv[top, 1], s=1, edgecolor='none', c=lecmaps.green())
ax.scatter(xy[bot, 0], uuv[bot, 1], s=1, edgecolor='none', c=lecmaps.violet())
title = r'$v=\partial_y \Phi$ at the surface'
ax.text(0.5, 1.05, title, ha='center', va='bottom', transform=ax.transAxes)
ax.set_xlabel(r'position, $x/L$')
ax.set_ylabel(r'normal velocity, $v=\mathbf{u}\cdot \mathbf{n}$')
plt.savefig(outdir + 'potential_flow_shell' + specstr + '_flowtop.pdf', dpi=300)
plt.close('all')
