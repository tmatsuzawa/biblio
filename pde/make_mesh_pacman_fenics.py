import numpy as np
import argparse
import matplotlib.pyplot as plt
import mesh_generation_xml_fenics as meshgen
import phasefield_fluids as pe
import lepm.stringformat as sf
import lepm.data_handling as dh
import lepm.math_functions as mf


'''Create a box with holes in it for pde solving, two piston outlets

Example usage:
python make_mesh_pacman_fenics.py -N 1000 -dshell 0.5 -thick 0.1 -phi 0.0
python make_mesh_pacman_fenics.py -N 1000 -dshell 0.5 -thick 0.1 -phi 0.25
python make_mesh_pacman_fenics.py -N 1000 -dshell 0.5 -thick 0.1 -phi 0.5
python make_mesh_pacman_fenics.py -N 1000 -dshell 0.5 -thick 0.1 -phi 1.5

python make_mesh_pacman_fenics.py -N 10000 -dshell 0.25 -thick 0.02 -phi 0.0
python make_mesh_pacman_fenics.py -N 10000 -dshell 0.25 -thick 0.02 -phi 0.25
python make_mesh_pacman_fenics.py -N 10000 -dshell 0.25 -thick 0.02 -phi 0.5
python make_mesh_pacman_fenics.py -N 10000 -dshell 0.25 -thick 0.02 -phi 1.5
'''

parser = argparse.ArgumentParser(description='Build mesh for use with fenics.')
parser.add_argument('-centeredy', '--centeredy', help='Put the shell in the center of the tank', action='store_true')
parser.add_argument('-dshell', '--dshell', help='Size of the shell', type=float, default=0.5)
parser.add_argument('-thole', '--theta_hole', help='Half size of the hole in radians', type=float, default=0.5)
parser.add_argument('-phi', '--phi', help='Angle at which center of hole points, in pi radians', type=float, default=0.)
parser.add_argument('-N', '--N', help='Number of points in the mesh (rough)', type=int, default=1000)
parser.add_argument('-R', '--R', help='Radius (or half-width) of the mesh', type=float, default=1.0)
parser.add_argument('-thick', '--thickness', help='Thickness of the shell', type=float, default=.10)
parser.add_argument('-LT', '--LatticeTop', help='Topology of mesh (Vogelmethod_Disc SquareLatt Triangular Trisel)',
                    type=str, default='Triangular')
parser.add_argument('-shape', '--shape', help='Shape of mesh (ex: square, circle, rectangle2x1, rectangle1x2)',
                    type=str, default='square')
parser.add_argument('-eta', '--eta', help='Jitter in lattice vertex positions, as fraction of lattice spacing',
                    type=float, default=0.0)
parser.add_argument('-theta', '--theta', help='Additional rotation of lattice vectors, as fraction of pi',
                    type=float, default=0.0)
parser.add_argument('-plot', '--force_plot', help='Whether to display resulting lattice in mpl', type=int,
                    default=0)
parser.add_argument('-outdir', '--outdir', help='Where to store mesh', type=str,
                    default='./meshes/')
args = parser.parse_args()

##############
# Parameters #
##############
thickness = args.thickness
rsz = args.R
nsz = args.N  # points
LatticeTop = args.LatticeTop  # ('Vogelmethod_Disc' 'SquareLatt' 'Triangular' 'Trisel')
shape = args.shape  # ('circle' 'square' 'rectangle2x1' 'rectangle1x2')
eta = args.eta  # randomization (jitter)
theta = args.theta * np.pi  # rotation of lattice vecs wrt x,y
force_plot = args.force_plot

#############
# Make mesh #
#############
# Create a box with interior boundaries
# Parameters
y1 = 0.05
dhole = 0.05
dpiston = 0.07
dshell = args.dshell

# Derived geometric variables
thole = args.theta_hole
specstr = '_dshell' + sf.float2pstr(dshell) + '_thetahole' + sf.float2pstr(thole) + \
          '_phihole' + sf.float2pstr(args.phi) + '_thick' + sf.float2pstr(args.thickness)

# Build the mesh
sz = np.ceil(np.sqrt(nsz))
latticevecs = [[1, 0], [0.5, np.sqrt(3) * 0.5]]
xy = pe.generate_lattice([2 * sz, 2 * sz], latticevecs) * rsz / (2 * sz)
xy -= np.array([np.min(xy[:, 0]), np.min(xy[:, 1])])
xy /= np.max(xy[:, 0]) - np.min(xy[:, 0])
extent = np.max(xy[:, 0]) - np.min(xy[:, 0])
yextent = np.max(xy[:, 1]) - np.min(xy[:, 1])

# Build the shell
tt = np.linspace(0, 2. * np.pi, 1000, endpoint=True)
tt = tt[np.where(np.logical_and(tt < 2. * np.pi - thole, tt > thole))[0]]
xshell, yshell = dshell * 0.5 * np.cos(tt), dshell * 0.5 * np.sin(tt)
inner_shell_x = (dshell * 0.5 - thickness) * np.cos(tt)
inner_shell_y = (dshell * 0.5 - thickness) * np.sin(-tt)
# Now append the two curves
xshell = np.hstack((xshell, inner_shell_x))
yshell = np.hstack((yshell, inner_shell_y))
# Rotate by phi
poly = np.dstack((xshell, yshell))[0]
poly = mf.rotate_vectors_2D(poly, args.phi * np.pi)
# Center the shell in the center of the mesh
xshell = poly[:, 0] + extent * 0.5
yshell = poly[:, 1] + yextent * 0.5
# Create rolled version for line segment definition
xshell1 = np.roll(xshell, 1)
yshell1 = np.roll(yshell, 1)
rmv_lsegs = np.dstack((xshell, yshell, xshell1, yshell1))[0]
polygon = np.dstack((xshell, yshell))[0]

# Check it
# print 'polygon = ', polygon
# plt.plot(polygon[:, 0], polygon[:, 1], 'b-')
# plt.show()

# Remove xy points from the shell area
xy = dh.pts_outside_polygon(xy, polygon)

add_params = {'polygon': polygon}

# Naming and output
fname = './meshes/shell_N' + str(args.N) + '_n' + str(len(xy)) + '_R{0:0.3f}'.format(extent).replace('.', 'p')
meshgen.generate_mesh_remove_lsegs(fname + specstr, xy, rmv_lsegs, thres=1.2, force_plot=True, add_params=add_params)
