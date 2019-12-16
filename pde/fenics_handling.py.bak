import numpy as np
import matplotlib.pyplot as plt
import os
import socket
import copy
import glob

try:
    import dolfin as dolf
except:
    print '\n\nWARNING: Could not import dolfin\n (Limits usage of module phasefield_elasticity).'
from matplotlib import cm

hostname = socket.gethostname()
if hostname[0:6] != 'midway':
    import scipy
    from scipy.interpolate import griddata

    try:
        import sympy as sp
    except:
        print 'WARNING: Could not import sympy!\n (Should not matter though).'
    from scipy.spatial import Delaunay

'''Module for handling fenics data.

Table of Contents
-----------------
1. FEniCS definitions
        (these functions generally use UFL)
        generating dolfin meshes from saved triangulations, defining boundary conditions
        initial phase conditions for cracks and interacting cracks
4. Lattice Generation
        generate a lattice given lattice vectors, create a Vogel-triangulated mesh (useful for triangulating disks)
        arrow function for making mesh arrows
5. Data Handling
        converting vectors and tensors from polar to cartesian, converting triangulations to bond lists,
        cutting bonds based on length, determining if points lie on lines or linesegments,
        calculating nearest points on a line segment or line, minimum distance from any linesegment in list, etc,
        creating unique-rowed arrays, matching orientations of triangulations, computing initial phase profile of a crack,
        kd trees, looking up values for a point based on proximity via a table,
        check if a variable is a number, rounding based on arbitrary thresholds, find a variable definition in a txt file


Dictionary of acronyms used in this doc
---------------------------------------
=======  ============
=======  ============
UFL      Unified Form Language, used by dolfin/FEniCS codes
BC       boundary condition
BCUP     whether a boundary condition is dirichlet (U for displacement) or natural (P for traction)
ECUML    Edge Crack Under Mixed Loading
nn       nearest neighbors
BL       bond list (uppercase means 2D)
bL       bond length (lowercase means 1D)
P        traction at boundary or Peclet number, depending on the context
PURL     boundary condition situation with essential BC on left side of a plate and natural BC on the right side
=======  ============

List of Predefined Boundary Conditions (options for string variable BCtype)
---------------------------------------------------------------------------
==========================   ============
For use with BCUP ==         'natural-essential':
==========================   ============
Usingleptcorner_Puniaxial    fix singlept in bottom left corner as u=(0,0), P is uniaxial
UsingleptcornerP*            fix singlept in bottom left corner as u=(0,0), P is followed by a string specified another
                             subconfiguration (ex, Puniaxial)
==========================   ============

==========================   ============
For use with BCUP ==         'essential' (applied to u) or 'natural' (applied to P, with U->E*U):
==========================   ============
uniaxialfree                 'U*(x[0]-xc)' for Vv.sub(0) --> constrain one dimension (x dim) on sides, also do DirichletBC(Vv.sub(1), Constant(0.0), boundary_singleptcorner)
uniaxialfreeX                constrain both dimensions on sides (free refers to the top and bottom)
uniaxialfreeY                constrain both dimensions on top and bottom (free refers to the left and right)
biaxial                      ('U*sqrt( pow(x[0]-xc,2)+pow(x[1]-yc, 2) )*cos(atan2(x[1]-yc,x[0]-xc))', 'U*sqrt( pow(x[0]-xc,2)+pow(x[1]-yc, 2) )*sin(atan2(x[1]-yc,x[0]-xc))')
fixleftX                     ('U*(x[0]-xc)' , '0.0') for bcu = dolf.DirichletBC(Vv, u_0, boundary_leftside),
uniaxial-PURL                ('U*(x[0]-xc)' , '0.0')
uniaxialDisc                 ('U*sqrt( pow(x[0]-xc,2)+pow(x[1]-yc, 2) )*cos(atan2(x[1]-yc,x[0]-xc))' ,'0.0')
==========================   ============

==========================   ============
For use with BCUP ==         'essential':
==========================   ============
uniaxialmixedfree_u1s1       ('U*(x[0]-xc)' , 'U*(x[0]-xc)')
uniaxialmixedfree_uvals1     ('val*U*(x[0]-xc)' , 'U*(x[0]-xc)')
uniaxialmixedfree_u1sval     ('U*(x[0]-xc)' , 'val*U*(x[0]-xc)')
free                         no constraint
fixbotY                      ('0.' ,'0.') along bottom edge
fixtopY                      ('0.' ,'0.') along top edge
fixbotcorner                 ('0.' ,'0.') just in the corner, one mesh triangle
==========================   ============

'''

##########################################
# 1. FEniCS definitions
##########################################


def xy_from_function_space(vf, uu, mesh):
    """Get coordinates in xy of the mesh using a vector function space (vf) and a field defined on that space (uu),
    along with the mesh itself

    Parameters
    ----------
    vf : vector function space or None
        a vector function space defined on a mesh
    """
    if vf is None:
        vf = dolf.FunctionSpace(mesh, "Lagrange", 1)
        uu = dolf.TrialFunction(vf)
    elif uu is None:
        uu = dolf.TrialFunction(vf)

    n = vf.dim()
    d = uu.geometric_dimension()
    dof_coordinates = vf.dofmap().tabulate_all_coordinates(mesh)
    dof_coordinates.resize((n, d))
    return dof_coordinates


def dolf_laplacian(f):
    """Using UFL, calc the laplacian of a scalar field"""
    return dolf.div(dolf.grad(f))


def genmesh(shape, meshtype, N, xi, theta, R, eta, fenicsdir='../'):
    """Load correct mesh from FEniCS/meshes/ directory, assuming we've used the mesh_generation_xml_fenics.py module
    to create the mesh already.
    """
    Rstr = '{0:.3f}'.format(R).replace('.', 'p')
    etastr = '{0:.3f}'.format(eta).replace('.', 'p')
    if shape == 'square':
        nx = int(np.sqrt(N))
        meshd = nx / (2 * R) * float(xi)
        if meshtype == 'UnitSquare':
            print 'Creating unit square mesh of ', meshtype, ' lattice topology...'
            mesh = dolf.UnitSquareMesh(nx, nx)
        else:
            print 'Creating square-shaped mesh of ', meshtype, ' lattice topology...'
            meshfile = fenicsdir + 'meshes/' + shape + 'Mesh_' + meshtype + '_eta' + etastr + '_R' + Rstr + '_N' + \
                       str(int(N)) + '.xml'
            mesh = dolf.Mesh(meshfile)
    elif shape == 'circle':
        print 'Creating circle-shaped mesh of ', meshtype, ' lattice topology...'
        meshd = 2 * np.sqrt(N / np.pi) * float(xi) / (2 * R)
        if meshtype == 'Trisel':
            add_exten = '_Nsp' + str(Nsp) + '_H' + '{0:.2f}'.format(H / R).replace('.', 'p') + \
                        '_Y' + '{0:.2f}'.format(Y / R).replace('.', 'p') + \
                        '_beta' + '{0:.2f}'.format(beta / np.pi).replace('.', 'p') + \
                        '_theta' + '{0:.2f}'.format(theta / np.pi).replace('.', 'p')
        else:
            add_exten = ''

        meshfile = fenicsdir + 'meshes/' + shape + 'Mesh_' + meshtype + add_exten + '_eta' + etastr + '_R' + Rstr + '_N' + str(
            int(N)) + '.xml'
        mesh = dolf.Mesh(meshfile)
    elif shape == 'rectangle2x1' or shape == 'rectangle1x2':
        print 'Creating circle-shaped mesh of ', meshtype, ' lattice topology...'
        meshd = np.sqrt(N * 0.5) * float(xi) / (2 * R)
        add_exten = ''

        meshfile = fenicsdir + 'meshes/' + shape + 'Mesh_' + meshtype + add_exten + '_eta' + etastr + '_R' + Rstr + '_N' + str(
            int(N)) + '.xml'
        print 'loading meshfile = ', meshfile
        mesh = dolf.Mesh(meshfile)

    print 'found meshfile = ', meshfile
    return mesh, meshd, meshfile


##########################################
# 4. Lattice Generation
##########################################
def generate_diskmesh(R, n, steps_azim):
    """Create an array of evenly spaced points in 2D on a disc using Vogel's method.

    Parameters
    ----------
    R : float
        radius of the disc
    n : int
        number of points within the disc, distributed by Vogel method
    steps_azim : int
        number of points on the boundary of the disc

    Returns
    ---------
    xypts : (n+steps_azim) x 2 array
        the positions of vertices on evenly distributed points on disc
    """
    # steps_azim is the azimuthal NUMBER of steps of the mesh
    # ----> NOT the step size/length
    # Note! The R value is INCLUSIVE!!

    # spiral pattern of points using Vogel's method--> Golden Triangles
    # The radius of the ith point is rho_i=R*sqrt(i/n)
    # n = 256
    radius = R * np.sqrt(np.arange(n) / float(n))
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)

    points = np.zeros((n, 2))
    points[:, 0] = np.cos(theta)
    points[:, 1] = np.sin(theta)
    points *= radius.reshape((n, 1))
    # plt.plot(points[:,0],points[:,1],'b.')

    vals = np.array([[R * np.cos(ii * 2 * pi / steps_azim), R * np.sin(ii * 2 * pi / steps_azim)] for ii in
                  np.arange(steps_azim)])  # circle points
    vals = np.reshape(vals, [-1, 2])  # [steps**2,2])

    xypts = np.vstack((points, vals))
    return xypts


def generate_diskmesh_vogelgap(R, n, steps_azim, fraction_edge_gap):
    """Create an array of evenly spaced points in 2D on a disc using Vogel's method, but only up to a smaller
    radius than the radius of the circle of points with angular density 2pi/steps_azim

    Parameters
    ----------
    R : float
        radius of the disc
    n : int
        number of points within the disc, distributed by Vogel method
    steps_azim : int
        number of points on the boundary of the disc
    fraction_edge_gap : float
        difference between Vogel radius and circle radius, as a fraction of radius R.

    Returns
    ---------
    xypts : (n+steps_azim) x 2 array
        the positions of vertices on evenly distributed points on disc
    """
    # This includes the Vogel method but only up to a smaller radius than the
    # radius of the circle of points with angular density 2pi/steps_azim. The
    # difference between Vogel radius and circle radius is given by
    # fraction_edge_gap, as a fraction of radius R.
    # steps_azim is the azimuthal NUMBER of steps of the mesh
    # ----> NOT the step size/length
    # Note! The R value is INCLUSIVE!!

    # spiral pattern of points using Vogel's method--> Golden Triangles
    # The radius of the ith point is rho_i=R*sqrt(i/n)
    # n = 256
    radius = R * (1 - fraction_edge_gap) * np.sqrt(np.arange(n) / float(n))
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)

    points = np.zeros((n, 2))
    points[:, 0] = np.cos(theta)
    points[:, 1] = np.sin(theta)
    points *= radius.reshape((n, 1))
    # plt.plot(points[:,0],points[:,1],'b.')

    vals = np.array([[R * np.cos(ii * 2 * np.pi / steps_azim), R * np.sin(ii * 2 * np.pi / steps_azim)] for ii in
                     np.arange(steps_azim)])  # circle points
    vals = np.reshape(vals, [-1, 2])  # [steps**2,2])

    xypts = np.vstack((points, vals))
    return xypts


def generate_lattice(image_shape, lattice_vectors):
    """Creates lattice of positions from arbitrary lattice vectors.

    Parameters
    ----------
    image_shape : 2 x 1 list (eg image_shape=[L,L])
        Width and height of the lattice (square)
    lattice_vectors : 2 x 1 list of 2 x 1 lists (eg [[1 ,0 ],[0.5,sqrt(3)/2 ]])
        The two lattice vectors defining the unit cell.

    Returns
    ----------
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    """
    # Generate lattice that lives in
    # center_pix = np.array(image_shape) // 2
    # Get the lower limit on the cell size.
    dx_cell = max(abs(lattice_vectors[0][0]), abs(lattice_vectors[1][0]))
    dy_cell = max(abs(lattice_vectors[0][1]), abs(lattice_vectors[1][1]))
    # Get an over estimate of how many cells across and up.
    nx = 2 * image_shape[0] // dx_cell
    ny = 2 * image_shape[1] // dy_cell
    # Generate a square lattice, with too many points.
    # Here I generate a factor of 8 more points than I need, which ensures
    # coverage for highly sheared lattices.  If your lattice is not highly
    # sheared, than you can generate fewer points.
    x_sq = np.arange(-nx, nx, dtype=float)
    y_sq = np.arange(-ny, nx, dtype=float)
    x_sq.shape = x_sq.shape + (1,)
    y_sq.shape = (1,) + y_sq.shape
    # Now shear the whole thing using the lattice vectors
    # transpose so that row is along x axis
    x_lattice = lattice_vectors[0][1] * x_sq + lattice_vectors[1][1] * y_sq
    y_lattice = lattice_vectors[0][0] * x_sq + lattice_vectors[1][0] * y_sq
    # Trim to fit in box.
    mask = ((x_lattice < image_shape[0] / 2.0)
            & (x_lattice > -image_shape[0] / 2.0))
    mask = mask & ((y_lattice < image_shape[1] / 2.0)
                   & (y_lattice > -image_shape[1] / 2.0))
    x_lattice = x_lattice[mask]
    y_lattice = y_lattice[mask]
    # Make output compatible with original version.
    out = np.empty((len(x_lattice), 2), dtype=float)
    out[:, 0] = y_lattice
    out[:, 1] = x_lattice
    i = np.lexsort((out[:, 1], out[:, 0]))  # sort primarily by x, then y
    xy = out[i]
    return xy


def arrow_mesh(x, y, z, dx, dy, dz, rotation_angle=0, tail_width=0.2, head_width=0.5, head_length=0.3, overhang=0.0):
    """Creates a mesh arrow (pts,tri) pointing from x,y,z to x+dx,y+dy,z+dz.

    Parameters
    ----------
    x,y,z : floats
        x,y,z position of the tail of the arrow
    dx,dy,dz : floats
        signed distances in x,y,z from the tail to the head of the arrow
    rotation_angle : float
        angle in radians by which arrow rotated about its long axis
    tail_width : float
        width of the arrow tail as fraction of arrow length (tail_width = |(1)-(7)| =|(2)-(6)| )
    head_width : float
        width of the arrow head as fraction of arrow length (head_width = |(3)-(5)|)
    head_length : float
        fraction of the arrow length that is part of the arrow head
    overhang : float
        fraction of the arrow length by which the pointy corners of the head extend behind the head
    """
    #            2|\
    # 0  _________| \
    #   |         1  \ 3
    #   |_________   /
    # 6         5 | /
    #           4 |/
    #
    # Begin by making arrow in the xy plane, with middle of tail at xy, pointing in x dir
    d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    pts0 = np.array([[0, d * tail_width * 0.5, 0], \
                     [d * (1. - head_length), d * tail_width * 0.5, 0], \
                     [d * (1. - head_length - overhang), d * head_width * 0.5, 0], \
                     [d, 0, 0], \
                     [d * (1. - head_length - overhang), -d * head_width * 0.5, 0], \
                     [d * (1. - head_length), -d * tail_width * 0.5, 0], \
                     [0, -d * tail_width * 0.5, 0] \
                     ])
    # Rotate about axis by rotation_angle
    pts = rotate_vector_xaxis3D(pts0, rotation_angle)
    # Rotate in xy plane
    theta = np.arctan2(dz, np.sqrt(dx ** 2 + dy ** 2))
    phi = np.arctan2(dy, dx)
    pts = rotate_vector_yaxis3D(pts, -theta)
    pts = rotate_vector_zaxis3D(pts, phi)
    pts += np.array([x, y, z])
    tri = np.array([[0, 6, 1], [6, 5, 1], \
                    [3, 2, 1], [4, 3, 5], [3, 1, 5]])
    return pts, tri


def rotate_vector_2D(vec, phi):
    """Rotate vector by angle phi in xy plane"""
    if vec.ndim > 1:
        '''rot is a list of multiple vectors or an array of length >1'''
        rot = np.array([[x * np.cos(phi) - y * np.sin(phi),
                         y * np.sin(phi) + y * np.cos(phi)] for x, y in vec])
    else:
        rot = np.array([vec[0] * np.cos(phi) - vec[1] * np.sin(phi),
                        vec[0] * np.sin(phi) + vec[1] * np.cos(phi)])
    return rot


####################################
# Rotations of arrays
####################################
def rotate_vector_xaxis3D(vec, phi):
    """Rotate 3D vector(s) by angle phi about x axis --> rotates away from the y axis"""
    if vec.ndim > 1:
        rot = np.array([[x,
                         y * np.cos(phi) - z * np.sin(phi),
                         y * np.sin(phi) + z * np.cos(phi)] for x, y, z in vec])
    else:
        rot = np.array([vec[0],
                        vec[1] * np.cos(phi) - vec[2] * np.sin(phi),
                        vec[1] * np.sin(phi) + vec[2] * np.cos(phi)])
    return rot


def rotate_vector_yaxis3D(vec, phi):
    """Rotate 3D vector(s) by angle phi about y axis (in xz plane) --> rotates away from the z axis"""
    if vec.ndim > 1:
        rot = np.array([[x * np.cos(phi) + z * np.sin(phi),
                         y,
                         -x * np.sin(phi) + z * np.cos(phi)] for x, y, z in vec])
    else:
        rot = np.array([vec[0] * np.cos(phi) + vec[2] * np.sin(phi),
                        vec[1],
                        -vec[0] * np.sin(phi) + vec[2] * np.cos(phi)])
    return rot


def rotate_vector_zaxis3D(vec, phi):
    """Rotate vector by angle phi in xy plane, keeping z value fixed"""
    if vec.ndim > 1:
        rot = np.array([[x * np.cos(phi) - y * np.sin(phi), \
                         x * np.sin(phi) + y * np.cos(phi), z] for x, y, z in vec])
    else:
        rot = np.array([vec[0] * np.cos(phi) - vec[1] * np.sin(phi), \
                        vec[0] * np.sin(phi) + vec[1] * np.cos(phi), vec[2]])
    return rot


##########################################
# 5. Data Handling
##########################################
def bond_length_list(xy, BL):
    """Convert bond list (#bonds x 2) to bond length list (#bonds x 1) for lattice of bonded points.

    Parameters
    ----------
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points.

    Returns
    ----------
    bL : array of dimension #bonds x 1
        Bond lengths, in order of BL (lowercase denotes 1D array)
    """
    bL = np.array(
        [np.sqrt(np.dot(xy[BL[i, 1], :] - xy[BL[i, 0], :], xy[BL[i, 1], :] - xy[BL[i, 0], :])) for i in range(len(BL))])
    return bL


def tensor_polar2cartesian2D(Mrr, Mrt, Mtr, Mtt, x, y):
    """converts a Polar tensor into a Cartesian one

    Parameters
    ----------
    Mrr, Mtt, Mrt, Mtr : N x 1 arrays
        radial, azimuthal, and shear components of the tensor M
    x : N x 1 array
        the x positions of the points on which M is defined
    y : N x 1 array
        the y positions of the points on which M is defined

    Returns
    ----------
    Mxx,Mxy,Myx,Myy : N x 1 arrays
        the cartesian components
    """
    A = Mrr;
    B = Mrt;
    C = Mtr;
    D = Mtt;
    theta = np.arctan2(y, x);
    ct = np.cos(theta);
    st = np.sin(theta);

    Mxx = ct * (A * ct - B * st) - st * (C * ct - D * st);
    Mxy = ct * (B * ct + A * st) - st * (D * ct + C * st);
    Myx = st * (A * ct - B * st) + ct * (C * ct - D * st);
    Myy = st * (B * ct + A * st) + ct * (D * ct + C * st);
    return Mxx, Mxy, Myx, Myy


def tensor_cartesian2polar2D(Mxx, Mxy, Myx, Myy, x, y):
    """converts a Cartesian tensor into a Polar one

    Parameters
    ----------
    Mxx,Mxy,Myx,Myy : N x 1 arrays
        cartesian components of the tensor M
    x : N x 1 array
        the x positions of the points on which M is defined
    y : N x 1 array
        the y positions of the points on which M is defined

    Returns
    ----------
    Mrr, Mrt, Mtr, Mtt : N x 1 arrays
        radial, shear, and azimuthal components of the tensor M
    """
    A = Mxx;
    B = Mxy;
    C = Myx;
    D = Myy;
    theta = np.arctan2(y, x);
    ct = np.cos(theta);
    st = np.sin(theta);

    Mrr = A * ct ** 2 + (B + C) * ct * st + D * st ** 2;
    Mrt = B * ct ** 2 + (-A + D) * ct * st - C * st ** 2;
    Mtr = C * ct ** 2 + (-A + D) * ct * st - B * st ** 2;
    Mtt = D * ct ** 2 - (B + C) * ct * st + A * st ** 2;
    return Mrr, Mrt, Mtr, Mtt


def vectorfield_cartesian2polar(ux, uy, x, y):
    """converts a Cartesian vector field into a Polar one

    Parameters
    ----------
    ux,uy : N x 1 arrays
        vector field values along x and y (cartesian)
    x : N x 1 array
        the x positions of the points on which u is defined
    y : N x 1 array
        the y positions of the points on which u is defined

    Returns
    ----------
    ur, ut : N x 1 arrays
        radial and azimuthal values of the vector field
    """
    theta = np.arctan2(y, x)
    ur = ux * np.cos(theta) + uy * np.sin(theta)
    ut = -ux * np.sin(theta) + uy * np.cos(theta)
    return ur, ut


def vectorfield_polar2cartesian(ur, ut, x, y):
    """converts a Polar vector field into a Cartesian one

    Parameters
    ----------
    ur,ut : N x 1 arrays
        vector field values along r and theta (polar)
    x : N x 1 array
        the x positions of the points on which u is defined
    y : N x 1 array
        the y positions of the points on which u is defined

    Returns
    ----------
    ux, uy : N x 1 arrays
        cartesian values of the vector field
    """
    theta = np.arctan2(y, x)
    beta = theta + np.arctan2(ut, ur)
    umag = np.sqrt(ur ** 2 + ut ** 2)
    ux = umag * np.cos(beta)
    uy = umag * np.sin(beta)
    return ux, uy


def flip_orientations_tris(TRI, xyz):
    """Flip triangulations such that their normals are facing upward"""
    for ii in range(len(TRI)):
        V = xyz[TRI[ii, 1], :] - xyz[TRI[ii, 0], :]
        W = xyz[TRI[ii, 2], :] - xyz[TRI[ii, 0], :]
        Nz = V[0] * W[1] - V[1] * W[0]
        if Nz < 0:
            temp = TRI[ii, 2]
            TRI[ii, 2] = TRI[ii, 0]
            TRI[ii, 0] = temp
    return TRI


def flip_all_orientations_tris(TRI):
    """Flip triangulations such that their normals are inverted"""
    temp = copy.deepcopy(TRI[:, 2])
    TRI[:, 2] = TRI[:, 0]
    TRI[:, 0] = temp
    return TRI


def Tri2BL(TRI):
    """Convert triangulation array (#tris x 3) to bond list (#bonds x 2) for 2D lattice of triangulated points.

    Parameters
    ----------
    TRI : array of dimension #tris x 3
        Each row contains indices of the 3 points lying at the vertices of the tri.

    Returns
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points"""
    BL1 = TRI[:, [0, 1]]
    BL2 = np.vstack((BL1, TRI[:, [0, 2]]))
    BL3 = np.vstack((BL2, TRI[:, [1, 2]]))
    BLt = np.sort(BL3, axis=1)
    BL = unique_rows(BLt)
    return BL


def BL2TRI(BL):
    """Convert bond list (#bonds x 2) to Triangulation array (#tris x 3) (using dictionaries for speedup and scaling)

    Parameters
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points

    Returns
    ----------
    TRI : array of dimension #tris x 3
        Each row contains indices of the 3 points lying at the vertices of the tri.
    """
    d = {}
    tri = np.zeros((len(BL), 3), dtype=np.int)
    c = 0
    for i in BL:
        if (i[0] > i[1]):
            t = i[0]
            i[0] = i[1]
            i[1] = t
        if (i[0] in d):
            d[i[0]].append(i[1])
        else:
            d[i[0]] = [i[1]]
    for key in d:
        for n in d[key]:
            for n2 in d[key]:
                if (n > n2) or n not in d:
                    continue
                if (n2 in d[n]):
                    tri[c, :] = [key, n, n2]
                    c += 1
    return tri[0:c]


def BL2NLandKL(BL, nn=6):
    """Convert bond list (#bonds x 2) to neighbor list (#pts x max# neighbors) for lattice of bonded points. Also returns KL: ones where there is a bond and zero where there is not.
    (Even if you just want NL from BL, you have to compute KL anyway.)

    Parameters
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    nn : int
        maximum number of neighbors

    Returns
    ----------
    NL : array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point.
    KL :  array of dimension #pts x (max number of neighbors)
        Spring constant list, where 1 corresponds to a true connection while 0 signifies that there is not a connection.
    """
    NL = np.zeros((max(BL.ravel()) + 1, nn))
    KL = np.zeros((max(BL.ravel()) + 1, nn))
    for row in BL:
        col = np.where(KL[row[0], :] == 0)[0][0]
        NL[row[0], col] = row[1]
        KL[row[0], col] = 1
        col = np.where(KL[row[1], :] == 0)[0][0]
        NL[row[1], col] = row[0]
        KL[row[1], col] = 1
    return NL, KL


def bond_length_list(xy, BL):
    """Convert neighbor list to bond list (#bonds x 2) for lattice of bonded points.

    Parameters
    ----------
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points.

    Returns
    ----------
    bL : array of dimension #bonds x 1
        Bond lengths, in order of BL (lowercase denotes 1D array)
    """
    bL = np.array(
        [np.sqrt(np.dot(xy[BL[i, 1], :] - xy[BL[i, 0], :], xy[BL[i, 1], :] - xy[BL[i, 0], :])) for i in range(len(BL))])
    return bL


def cut_bonds(BL, xy, thres):
    """Cuts bonds with lengths greater than threshold value.

    Parameters
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    thres : float
        cutoff length between points

    Returns
    ----------
    BLtrim : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points, contains no bonds longer than thres"""
    i2cut = (xy[BL[:, 0], 0] - xy[BL[:, 1], 0]) ** 2 + (xy[BL[:, 0], 1] - xy[BL[:, 1], 1]) ** 2 < thres ** 2
    BLtrim = BL[i2cut]
    return BLtrim


def memberIDs(a, b):
    """Return array (c) of indices where elements of a are members of b.
    If ith a elem is member of b, ith elem of c is index of b where a[i] = b[index].
    If ith a elem is not a member of b, ith element of c is 'None'.
    The speed is O(len(a)+len(b)), so it's fast.
    """
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]  # None can be replaced by any other "not in b" value


def ismember(a, b):
    """Return logical array (c) testing where elements of a are members of b.
    The speed is O(len(a)+len(b)), so it's fast.
    """
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = True
    return np.array([bind.get(itm, False) for itm in a])  # None can be replaced by any other "not in b" value


def unique_rows(a):
    """Clean up an array such that all its rows are unique.
    """
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]


def do_kdtree(combined_x_y_arrays, points, k=1):
    """Using kd tree, return indices of nearest points and their distances

    Parameters
    ----------
    combined_x_y_arrays : NxD array
        the reference points of which to find nearest ones to 'points' data
    points : MxD array
        data points, finds nearest elements in combined_x_y_arrays to these points.

    Returns
    ----------
    indices : Mx1 array
        indices of xyref that are nearest to points
    dist : Mx1 array
        the distances of xyref[indices] from points
    """
    # usage--> find nearest neighboring point in combined_x_y_arrays for
    # each point in points.
    # Note: usage for KDTree.query(x, k=1, eps=0, p=2, distance_upper_bound=inf)[source]
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points, k=k)
    return indexes, dist


def lookupZ(lookupXYZ, xy_pts):
    """Using kd tree, convert array of xy points to xyz points by lookup up Z values, (based on proximity of xy pts).
    See also lookupZ_avgN().

    Parameters
    ----------
    lookupXYZ : Nx3 array
        the reference points of which to find nearest ones to 'points' data
    xy_pts : MxD array with D>=2
        data points, finds nearest elements in combined_x_y_arrays to these points.

    Returns
    ----------
    outXYZpts : Nx3 array
        x,y are from xy_pts, but z values from lookupXYZ
    """
    # print 'xy_pts = ', xy_pts
    # print 'with shape ', np.shape(xy_pts)
    Xtemp = lookupXYZ[:, 0]
    Ytemp = lookupXYZ[:, 1]
    lookupXY = np.dstack([Xtemp.ravel(), Ytemp.ravel()])[0]
    # Find addZ, the amount to raise the xy_pts in z.
    addZind, distance = do_kdtree(lookupXY, xy_pts)
    addZ = lookupXYZ[addZind, 2]
    # print 'addZ = ', addZ
    # print 'with shape ', np.shape(addZ.ravel())
    x = np.ravel(xy_pts[:, 0])
    y = np.ravel(xy_pts[:, 1])
    # print 'shape of x = ', np.shape(x.ravel())
    outXYZpts = np.dstack([x.ravel(), y.ravel(), addZ.ravel()])[0]
    # View output
    # fig = plt.figure(figsize=(14,6))
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # scatter(outXYZpts[:,0],outXYZpts[:,1],outXYZpts[:,2],c='b')

    return outXYZpts


def lookupZ_avgN(lookupXYZ, xy_pts, N=5, method='median'):
    """Using kd tree, return array of values for xy_pts given lookupXYZ based on near neighbors.
    Average over N neighbors for the returned value.

    Parameters
    ----------
    lookupXYZ : Nx3 array
        the reference points of which to find nearest ones to 'points' data
    xy_pts : MxD array with D>=2
        data points, finds nearest elements in combined_x_y_arrays to these points.
    N : int
        number of nearby particles over which to average in the lookup evaluation

    Returns
    ----------
    outXYZpts : Nx3 array
        x,y are from xy_pts, but z values from lookupXYZ
    """
    # print 'xy_pts = ', xy_pts
    # print 'with shape ', np.shape(xy_pts)
    Xtemp = lookupXYZ[:, 0]
    Ytemp = lookupXYZ[:, 1]
    lookupXY = np.dstack([Xtemp.ravel(), Ytemp.ravel()])[0]
    # Find addZ, the amount to raise the xy_pts in z.
    addZind, distance = do_kdtree(lookupXY, xy_pts, k=N)
    # print 'addZind =', addZind
    if isinstance(lookupXYZ, np.ma.core.MaskedArray):
        lookupXYZ = lookupXYZ.data
    if method == 'median':
        addZ = np.array([[np.median(lookupXYZ[addZind[ii, :], 2])] for ii in range(len(addZind))])
    elif method == 'mean':
        addZ = np.array([[np.nanmean(lookupXYZ[addZind[ii, :], 2])] for ii in range(len(addZind))])

    # print 'addZ = ', addZ
    # print 'with shape ', np.shape(addZ.ravel())
    x = np.ravel(xy_pts[:, 0])
    y = np.ravel(xy_pts[:, 1])
    # print 'shape of x = ', np.shape(x.ravel())
    outXYZpts = np.dstack([x.ravel(), y.ravel(), addZ.ravel()])[0]
    # View output
    # fig = plt.figure(figsize=(14,6))
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # scatter(outXYZpts[:,0],outXYZpts[:,1],outXYZpts[:,2],c='b')

    return outXYZpts


def lookupZ_singlept(lookupXYZ, xy):
    """Using kd tree, return indices of nearest points and their distances using 2D positions.

    Parameters
    ----------
    lookupXYZ : Nx3 array
        the reference points of which to find nearest ones to 'points' data
    xy : list of two floats
        data point, to find nearest element in lookupXYZ

    Returns
    ----------
    addZ : float
        z value from lookupXYZ
    """
    Xtemp = lookupXYZ[:, 0]
    Ytemp = lookupXYZ[:, 1]
    lookupXY = np.dstack([Xtemp.ravel(), Ytemp.ravel()])[0]
    # Find addZ, the amount to raise the xy_pts in z.
    addZind, distance = do_kdtree(lookupXY, np.array([xy[0], xy[1]]))
    addZ = lookupXYZ[addZind, 2]
    return addZ

def is_number(s):
    """Check if a string can be represented as a number; works for floats"""
    try:
        float(s)
        return True
    except ValueError:
        return False


def round_thres(a, MinClip):
    """Round a number to the nearest multiple of MinCLip"""
    return round(float(a) / MinClip) * MinClip


def round_thres_numpy(a, MinClip):
    """Round an array of values to the nearest multiple of MinCLip"""
    return np.round(np.array(a, dtype=float) / MinClip) * MinClip


def getVarFromFile(filename):
    """Convert data in a txt file like 'x = 1.5' to a variable x defined as 1.5... this may need work
    http://stackoverflow.com/questions/924700/best-way-to-retrieve-variable-values-from-a-text-file-python-json
    """
    import imp
    f = open(filename)
    data = imp.load_source('data', '', f)
    f.close()
    return data


##########################################
# 6. Loading/Interpolating Data
##########################################
def nearest_gL_fit(lookupdir, beta, rho, fit_mean):
    """Lookup Griffith length for given rho value in table, could be table based on a quadratic fit or of the mean gLs for a given rho.
    Note that for fit_mean==fit, rho = r/R, wherease for fit_mean==mean, rho = r/x0."""
    print('looking for file:')
    print(lookupdir + fit_mean + '_rho_gLmeters_beta' + '{0:.2f}'.format(beta / np.pi).replace('.', 'p') + '*.txt')
    gLfile = \
        glob.glob(
            lookupdir + fit_mean + '_rho_gLmeters_beta' + '{0:.2f}'.format(beta / np.pi).replace('.', 'p') + '*.txt')[
            0]
    rhoV, gLV, trsh = np.loadtxt(gLfile, delimiter=',', skiprows=1, usecols=(0, 1, 2), unpack=True)
    diff = abs(rhoV - rho)
    IND = np.where(diff == diff.min())
    return gLV[IND][0]


def constP_gL_fit(lookupdir, alph):
    """Lookup Griffith length for given aspect ratio in table of the mean gLs vs aspect ratio, returned in meters"""
    print('looking for file:')
    print(lookupdir + 'constP_means_alph_gLinches.txt')
    gLfile = glob.glob(lookupdir + 'constP_means_alph_gLinches.txt')[0]
    alphV, gLV = np.loadtxt(gLfile, delimiter=',', skiprows=1, usecols=(0, 1), unpack=True)
    diff = abs(alphV - alph)
    IND = np.where(diff == diff.min())
    # return in meters
    return float(gLV[IND] / 39.3700787)


def interpol_meshgrid(x, y, z, n):
    """Interpolate z on irregular or unordered grid data (x,y) by supplying # points along each dimension.
    Note that this does not guarantee a square mesh, if ranges of x and y differ.
    """
    # define regular grid spatially covering input data
    xg = np.linspace(x.min(), x.max(), n)
    yg = np.linspace(y.min(), y.max(), n)
    X, Y = np.meshgrid(xg, yg)

    # interpolate Z values on defined grid
    Z = griddata(np.vstack((x.flatten(), y.flatten())).T, np.vstack(z.flatten()), (X, Y), method='cubic').reshape(
        X.shape)
    # mask nan values, so they will not appear on plot
    Zm = np.ma.masked_where(np.isnan(Z), Z)
    return X, Y, Zm


def interpolate_onto_mesh(x, y, z, X, Y, mask=True):
    """Interpolate new data x,y,z onto grid data X,Y"""
    # interpolate Z values on defined grid
    Z = griddata(np.vstack((x.flatten(), y.flatten())).T, np.vstack(z.flatten()), (X, Y), method='cubic').reshape(
        X.shape)
    # mask nan values, so they will not appear on plot
    if mask:
        Zm = np.ma.masked_where(np.isnan(Z), Z)
    else:
        Zm = Z
    return Zm


##########################################
# Files, Folders, and Directory Structure
##########################################
def prepdir(dir):
    """Make sure that the (string) variable dir ends with the character '/'.
    This prepares the string dir to be an output directory."""
    if dir[-1] == '/':
        return dir
    else:
        return dir + '/'


def ensure_dir(f):
    """Check if directory exists, and make it if not.

    Parameters
    ----------
    f : string
        directory path to ensure

    Returns
    ----------
    """
    f = prepdir(f)
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


def find_dir_with_name(name, searchdir):
    """Return a path or list of paths to directories which match the string 'name' (can have wildcards) in searchdir.
    Note that this function returns names with a trailing back slash (/)"""
    if name == '':
        '''no name given, no name returned'''
        return []
    else:
        possible_dirs = glob.glob(searchdir + name)
        okdirs = [os.path.isdir(possible_dir) for possible_dir in possible_dirs]
        out = [possible_dirs[i] + '/' for i in range(len(okdirs)) if okdirs[i]]
        if len(out) == 1:
            return out[0]
        else:
            return out


def find_subdirs(string, maindir):
    """Find subdir(s) matching string, in maindir. Return subdirs as list.
    If there are multiple matching subdirectories, returns list of strings.
    If there are no matches, returns empty list.
    """
    maindir = prepdir(maindir)
    contents = sorted(glob.glob(maindir + string))
    is_subdir = [os.path.isdir(ii) for ii in contents]

    if len(is_subdir) == 0:
        print 'WARNING! Found no matching subdirectory: returning empty list'
        return is_subdir
    else:
        subdirs = [prepdir(contents[ii]) for ii in np.where(is_subdir)[0].tolist()]

    return subdirs


def find_subsubdirectory(string, maindir):
    """Find subsubdir matching string, in maindir. Return subdir and subsubdir names.
    If there are multiple matching subdirectories, returns list of strings.
    If there are no matches, returns empty lists.
    """
    maindir = prepdir(maindir)
    # print 'maindir = ', maindir
    contents = glob.glob(maindir + '*')
    is_subdir = [os.path.isdir(ii) for ii in contents]

    if len(is_subdir) == 0:
        print 'WARNING! Found no matching subdirectory: returning empty list'
        return is_subdir, is_subdir
    else:
        # print 'contents = ', contents
        subdirs = [contents[ii] for ii in np.where(is_subdir)[0].tolist()]
        # print 'subdirs = ', subdirs

    found = False
    subsubdir = []
    for ii in subdirs:
        # print 'ii =', ii
        print 'prepdir(ii)+string = ', prepdir(ii) + string
        subcontents = glob.glob(prepdir(ii) + string)
        # print 'glob.glob(',prepdir(ii),string,') = ',subcontents
        is_subsubdir = [os.path.isdir(jj) for jj in subcontents]
        subsubdirs = [subcontents[jj] for jj in np.where(is_subsubdir)[0].tolist()]
        # print 'subsubdirs = ', subsubdirs
        if len(subsubdirs) > 0:
            if found == False:
                if len(subsubdirs) == 1:
                    subdir = prepdir(ii)
                    subsubdir = prepdir(subsubdirs[0])
                    # print 'adding first subdir = ', subdir
                    found = True
                elif len(subsubdirs) > 1:
                    subdir = [prepdir(ii)] * len(subsubdirs)
                    # print 'adding first few subdir = ', subdir
                    found = True
                    subsubdir = [0] * len(subsubdirs)
                    for j in range(len(subsubdirs)):
                        subsubdir[j] = prepdir(subsubdirs[j])
            else:
                # Since already found one, add another
                # print ' Found more subsubdirs'
                # print 'subdir = ', subdir

                # Add subdir to list
                if isinstance(subdir, str):
                    subdir = [subdir, prepdir(ii)]
                    print 'adding second to subdir = ', subdir
                    if len(subsubdirs) > 1:
                        for kk in range(1, len(subsubdirs)):
                            subdir.append(prepdir(ii))
                        print 'adding second (multiple) to subdir = ', subdir
                else:
                    print 'subsubdirs'
                    for kk in range(1, len(subsubdirs)):
                        subdir.append(prepdir(ii))
                        # print 'subsubdirs = ', subsubdirs
                        print 'adding more to subdir = ', subdir
                # Add subsubdir to list
                for jj in subsubdirs:
                    if isinstance(subsubdir, str):
                        subsubdir = [subsubdir, prepdir(jj)]
                        print 'adding second to subsubdirs = ', subsubdir
                    else:
                        subsubdir.append(prepdir(jj))
                        print 'adding more to subsubdirs = ', subsubdir

    if found:
        return subdir, subsubdir
    else:
        return '', ''


##########################################
# Specific Geometric Setups
##########################################

##########################################
# A. Inclined Crack in Uniaxial loading
##########################################

def ICUL_kink_angle(beta):
    """Compute kink angle for an Inclined Crack in a Uniaxially Loaded plate (ICUL)"""
    eta = np.cos(beta) / np.sin(beta)
    kink = -2 * np.arctan(2 * eta / (1 + np.sqrt(1 + 8. * eta ** 2)))
    return kink


##########################################
# B. Quenched Glass Plate (QGP)
##########################################

def Z_Ttube_approx(x, y, decayL=0.2, DT=1.0, P=7.9, coldL=0.0, totLen=0.12, minY=0.0, L=0.12,
                   polyorder='Quartic4_2xW'):
    """Project x,y pts to surface of cylinder which narrows in a manner that approximates the curvature
    distribution of the glass plate in the limit of one radius of curvature (that of the tube) nearly constant.
    Some arguments are required for maximum efficiency, such as totLen = max(y)-min(y).

    Parameters
    ----------
    decayL : fraction of L that is used for decay
    totLen : height of sample, could be 4*R, for instance
    L : 2*R --> width, also radius of cylinder in cold region

    """
    # First do zi and pi
    z0 = np.amin(np.dstack((L * np.ones(len(x)), L - L * DT * (1 - np.exp(-P * (y - minY - coldL * totLen)))))[0],
                 axis=1)
    cL = coldL * totLen
    dL = decayL * L
    if polyorder == 'Quartic4_2xW':
        # negative y inds
        zi = y < minY + cL - dL
        ni = np.logical_and(y < minY + cL + dL, y > minY + cL - dL)
        pi = y > minY + cL + dL
        # print 'len(y)=', len(y)
        # print zi
        # print 'len(zi)=', np.where(zi)
        # print ni
        # print 'len(ni)=', len(np.where(ni))
        # print pi
        # print 'len(pi)=', len(np.where(pi==True))
        # Replace Heaviside with 3-7th order polynomial
        d = dL
        A = - np.exp(-P * d) * DT * L * (-105. +
                                         105. * np.exp(P * d) -
                                         90. * P * d -
                                         30. * P ** 2 * d ** 2 -
                                         4. * P ** 3 * d ** 3) / (48. * d ** 4)
        B = np.exp(-P * d) * DT * L * (-42. +
                                       42. * np.exp(P * d) -
                                       39. * P * d -
                                       14. * P ** 2 * d ** 2 -
                                       2. * P ** 3 * d ** 3) / (16. * d ** 5)
        C = - np.exp(-P * d) * DT * L * (-35. +
                                         35 * np.exp(P * d) -
                                         34 * P * d -
                                         13 * P ** 2 * d ** 2 -
                                         2 * P ** 3 * d ** 3) / (32. * d ** 6)
        D = np.exp(-P * d) * DT * L * (-15. +
                                       15. * np.exp(P * d) -
                                       15. * P * d - 6. * P ** 2 * d ** 2 -
                                       P ** 3 * d ** 3) / (96. * d ** 7)
        # Offset y --> yni by dL so that effectively polynomial is funct of 2*epsilon (ie 2*decayL)
        yni = y[ni] - minY - cL + dL
        z0[
            ni] = L + A * yni ** 4 + B * yni ** 5 + C * yni ** 6 + D * yni ** 7  # Heaviside --> make this quickly decaying polynomial

    f = z0 * np.cos(x / z0)
    return f


def Z_Ttube_approx_flipy(x, y, decayL=0.2, DT=1.0, P=7.9, coldL=0.0, totLen=0.12, minY=0.0, L=0.12,
                         polyorder='Quartic4_2xW'):
    """Project x,y pts to surface of cylinder which narrows in a manner that approximates the curvature
    distribution of the glass plate in the limit of one radius of curvature (that of the tube) nearly constant.
    Some arguments are required for maximum efficiency, such as totLen = max(y)-min(y).

    Parameters
    ----------
    decayL : fraction of L that is used for decay
    totLen : height of sample, could be 4*R, for instance
    L : 2*R --> width, also radius of cylinder in cold region

    """
    # First do zi and pi
    z0 = np.amin(np.dstack((L * np.ones(len(x)), L - L * DT * (1 - np.exp(-P * (-y + minY + coldL * totLen)))))[0],
                 axis=1)
    cL = coldL * totLen
    dL = decayL * L
    if polyorder == 'Quartic4_2xW':
        # negative y inds
        zi = y < minY + cL - dL
        ni = np.logical_and(y < minY + cL + dL, y > minY + cL - dL)
        pi = y > minY + cL + dL
        # print 'len(y)=', len(y)
        # print zi
        # print 'len(zi)=', np.where(zi)
        # print ni
        # print 'len(ni)=', len(np.where(ni))
        # print pi
        # print 'len(pi)=', len(np.where(pi==True))
        # Replace Heaviside with 3-7th order polynomial
        d = dL
        A = - np.exp(-P * d) * DT * L * (-105. +
                                         105. * np.exp(P * d) -
                                         90. * P * d -
                                         30. * P ** 2 * d ** 2 -
                                         4. * P ** 3 * d ** 3) / (48. * d ** 4)
        B = np.exp(-P * d) * DT * L * (-42. +
                                       42. * np.exp(P * d) -
                                       39. * P * d -
                                       14. * P ** 2 * d ** 2 -
                                       2. * P ** 3 * d ** 3) / (16. * d ** 5)
        C = - np.exp(-P * d) * DT * L * (-35. +
                                         35 * np.exp(P * d) -
                                         34 * P * d -
                                         13 * P ** 2 * d ** 2 -
                                         2 * P ** 3 * d ** 3) / (32. * d ** 6)
        D = np.exp(-P * d) * DT * L * (-15. +
                                       15. * np.exp(P * d) -
                                       15. * P * d - 6. * P ** 2 * d ** 2 -
                                       P ** 3 * d ** 3) / (96. * d ** 7)
        # Offset y --> yni by dL so that effectively polynomial is funct of 2*epsilon (ie 2*decayL)
        yni = y[ni] - minY - cL + dL
        # Heaviside --> make this quickly decaying polynomial
        z0[ni] = L + A * yni ** 4 + B * yni ** 5 + C * yni ** 6 + D * yni ** 7

    f = z0 * np.cos(x / z0)
    return f


def Ktinterp(ylin, coldL=2.0, decayL=0.2, alph=1.0, P=7.9, Lscale=1.0, polyorder='Quartic4_2xW'):
    """Return an interpolation of the target curvature for a temperature profile in a QGP.
    Let y=0 be the base of the strip. Usually distances are measured in units of strip halfwidth.

    Parameters
    ----------
    ylin : Nx1 array
        linspace over which to interpolate the curvature; must be evenly spaced
    alph : float (default = 1.0)
        coefficient of thermal expansion (overall scaling of G)
    P : float (default = 7.9)
        Peclet number = b*v/D  (halfwidth x velocity / coefficient of thermal diffusion)
    Lscale : float
        Length scale of the half strip width in other units. For ex, in units of the radius of curv of a surface
    """
    xs = sp.Symbol('xs')
    Ts = (1. - sp.exp(-P * (xs - coldL)))
    fTs = sp.lambdify(xs, Ts, 'numpy')
    dy = ylin[2] - ylin[1]  # grab one of the difference values --> must all be the same
    T = fTs(ylin)
    if polyorder == 'Quartic4':
        # negative y inds
        zi = ylin < coldL
        ni = np.logical_and(ylin < coldL + decayL, ylin > coldL)
        pi = ylin > coldL + decayL
        # Replace Heaviside with 3-7th order polynomial
        d = decayL
        A = np.exp(-P * d) * (-210. + 210. * np.exp(P * d) - 90. * P * d - 15. * P ** 2 * d ** 2 - P ** 3 * d ** 3) / (
            6. * d ** 4)
        B = - np.exp(-P * d) * (
            -168. + 168. * np.exp(P * d) - 78. * P * d - 14. * P ** 2 * d ** 2 - P ** 3 * d ** 3) / (2. * d ** 5)
        C = np.exp(-P * d) * (-140. + 140. * np.exp(P * d) - 68. * P * d - 13. * P ** 2 * d ** 2 - P ** 3 * d ** 3) / (
            2. * d ** 6)
        D = - np.exp(-P * d) * (
            -120. + 120. * np.exp(P * d) - 60. * P * d - 12. * P ** 2 * d ** 2 - P ** 3 * d ** 3) / (6. * d ** 7)
        y = ylin[ni] - coldL
        T[ni] = A * y ** 4 + B * y ** 5 + C * y ** 6 + D * y ** 7  # Heaviside --> make this quickly decaying polynomial
    elif polyorder == 'Quartic4_2xW':
        # negative y inds
        zi = ylin < coldL - decayL
        ni = np.logical_and(ylin < coldL + decayL, ylin > coldL - decayL)
        pi = ylin > coldL + decayL
        # Replace Heaviside with 3-7th order polynomial
        d = decayL
        A = np.exp(-P * d) * (
            -105. + 105. * np.exp(P * d) - 90. * P * d - 30. * P ** 2 * d ** 2 - 4. * P ** 3 * d ** 3) / (48. * d ** 4)
        B = - np.exp(-P * d) * (
            - 42. + 42. * np.exp(P * d) - 39. * P * d - 14. * P ** 2 * d ** 2 - 2. * P ** 3 * d ** 3) / (16. * d ** 5)
        C = np.exp(-P * d) * (
            - 35. + 35. * np.exp(P * d) - 34. * P * d - 13. * P ** 2 * d ** 2 - 2. * P ** 3 * d ** 3) / (32. * d ** 6)
        D = - np.exp(-P * d) * (
            - 15. + 15. * np.exp(P * d) - 15. * P * d - 6. * P ** 2 * d ** 2 - 1. * P ** 3 * d ** 3) / (96. * d ** 7)
        y = ylin[ni] - coldL + decayL
        T[ni] = A * y ** 4 + B * y ** 5 + C * y ** 6 + D * y ** 7  # Heaviside --> make this quickly decaying polynomial

    T[zi] = 0.
    Ty = np.gradient(T, dy)
    Tyy = np.gradient(Ty, dy)
    # plt.plot(ylin, T, 'k.', label='T')
    # plt.plot(ylin, Ty, 'g.', label='Ty')
    # plt.plot(ylin, Tyy, 'b.', label='Tyy')
    # plt.legend()
    # plt.title(r'$T$, $\partial_y T$, $\partial_y^2 T$')
    # plt.show()
    Ktinterp = scipy.interpolate.interp1d(ylin, alph * Tyy / Lscale ** 2)
    return Ktinterp


########
# DEMO #
########

if __name__ == "__main__":
    demo_arrow_mesh = False
    demo_tensor = False
    demo_vectfield = False
    demo_linept = False
    demo_gaussiancurvature = False
    demo_gaussiancurvature2 = False
    demo_Ztube = False
    demo_initial_phase_multicrack = False
    demo_GB_elastic_theory = True

    ptsz = 50  # size of dot for scatterplots
    from mpl_toolkits.mplot3d import Axes3D

    if demo_arrow_mesh:
        print 'Demonstrating arrow_mesh function: makes custom arrows in 3D'
        fig = plt.figure()  # figsize=plt.figaspect(1.0))
        ax = fig.gca(projection='3d')
        x = .5;
        y = .5;
        z = 1.0;
        # Make a bunch of arrows
        p0, t0 = arrow_mesh(x, y, 0, 1, 0, 0)
        p1, t1 = arrow_mesh(x, y, 0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2), rotation_angle=0.0 * np.pi)
        p2, t2 = arrow_mesh(x, y, 0, 1, 0, 0, rotation_angle=0.0 * np.pi)
        p3, t3 = arrow_mesh(x, y, z, 1.0, 0.5, 0.5, rotation_angle=0.0 * np.pi, head_length=0.1)
        p4, t4 = arrow_mesh(x, y, z, 0.1, -0.5, -0.3, rotation_angle=0.25 * np.pi, overhang=0.1)
        p5, t5 = arrow_mesh(x, y, z, 0.0, 0.5, -0.5, rotation_angle=0.5 * np.pi, tail_width=0.1, overhang=0.2)
        p6, t6 = arrow_mesh(x, y, z, 0.2, 0.3, 0.5, rotation_angle=0.75 * np.pi, head_width=0.8, overhang=0.1)
        p = [p0, p1, p2, p3, p4, p5, p6]
        for ii in range(len(p)):
            pi = p[ii]
            ax.plot_trisurf(pi[:, 0], pi[:, 1], pi[:, 2], triangles=t0, cmap=cm.jet)
        ax.set_xlabel('x');
        ax.set_ylabel('y');
        ax.set_zlabel('z')
        plt.show()
        plt.close('all')

    if demo_tensor:
        # Show gaussian bump stress
        print('Demonstrating calculation and easy display of Stress field for Gaussian Bump and ' +
              'conversion to Cartesian coords')
        x = np.linspace(-5, 5)
        y = np.linspace(-5, 5)
        xv, yv = np.meshgrid(x, y)
        x = xv.ravel()
        y = yv.ravel()
        t = np.sqrt(x ** 2 + y ** 2)

        alph = 0.3
        x0 = 1.5
        R = 5
        U = 0.0
        nu = 0.4
        Srr, Stt = GB_stress_uDirichlet(alph, x0, R, U, nu, t)
        Srt = np.zeros_like(Srr)
        Str = np.zeros_like(Srr)

        Sxx, Sxy, Syx, Syy = tensor_polar2cartesian2D(Srr, Srt, Str, Stt, x, y)

        # Polar version
        title0 = r'Polar Stresses for Bump: $x_0$=' + str(x0) + r' $\alpha$=' + str(alph) + r' $U=$' + str(U) + \
                 r' $R=$' + str(R)
        pf_display_tensor(x, y, Srr, Srt, Str, Stt, r'\sigma', title=title0, subscripts='polar', ptsz=20, axis_on=0)

        # Cartesian version
        title0 = r'Cartesian Stresses for Bump: $x_0$=' + str(x0) + r' $\alpha$=' + str(alph) + r' $U=$' + str(
            U) + r' $R=$' + str(R)
        pf_display_tensor(x, y, Sxx, Sxy, Syx, Syy, r'\sigma', title=title0, subscripts='cartesian', ptsz=20, axis_on=0)

    if demo_vectfield:
        # Show gaussian bump displacement in r,theta
        print('Demonstrating calculation and easy display of displacement field for Gaussian Bump and sinusoidal field')
        x = np.linspace(-5, 5)
        y = np.linspace(-5, 5)
        xv, yv = np.meshgrid(x, y)
        x = xv.ravel()
        y = yv.ravel()
        t = np.sqrt(x ** 2 + y ** 2)

        alph = 0.3;
        x0 = 1.5;
        R = 5;
        U = 0.02;
        nu = 0.4;
        ur = GB_displacement_uDirichlet(alph, x0, R, U, nu, t)
        ut = np.zeros_like(ur)

        varchar = r'u'
        title0 = r'Bump: $x_0$=' + str(x0) + r' $\alpha$=' + str(alph) + r' $U=$' + str(U) + r' $R=$' + str(R)
        pf_display_vector(x, y, ur, ut, varchar, title=title0, subscripts='polar', ptsz=20, axis_on=0)

        # Show conversion from displacement field to polar coords
        print('Demonstrating conversion from cartesian displacement field to polar coords and vice versa')
        ux0 = np.cos(x)
        uy0 = np.cos(y)
        ur, ut = vectorfield_cartesian2polar(ux0, uy0, x, y)
        ux, uy = vectorfield_polar2cartesian(ur, ut, x, y)
        pf_display_4panel(x, y, ur, ut, ux, uy, r'Sine field $u_r$', title1=r'Sine field $u_\theta$', \
                          title2=r'Sine field $u_x$', title3=r'Sine field $u_y$', ptsz=20, axis_on=0)

    if demo_linept:
        print 'Demo: Define value based on distance from a line segment (used for creating initial state of phase for a crack).'
        print 'This demo focuses on a function that does this for one point at a time (inefficient in numpy but useful in some contexts in FEniCS).'
        pts = np.random.random((4000, 2))
        W = .3
        endpt1 = [W, W]
        endpt2 = [1. - W, 1. - W]

        value = np.zeros_like(pts[:, 0])
        ind = 0
        for pt in pts:
            print 'pt=', pt
            x = [pt[0], pt[1]]
            print 'x=', x
            # p, d = closest_pt_on_lineseg(x,endpt1, endpt2)
            value[ind] = initphase_linear_slit(x, endpt1, endpt2, W, contour='linear')
            ind += 1
        pf_display_scalar(pts[:, 0], pts[:, 1], value, 'Phase values near a crack', ptsz=40, axis_on=0)
        plt.show()

    if demo_gaussiancurvature:
        print 'Demo: Demonstrating gaussian_curvature_unstructured: measuring the Gaussian curvature of a surface defined by a collection of points'
        X = np.arange(-5, 5, 0.2)
        Y = np.arange(-5, 5, 0.2)
        X, Y = np.meshgrid(X, Y)
        X = X.ravel()
        Y = Y.ravel()
        R = np.sqrt(X ** 2 + Y ** 2)
        Z = np.exp(-R ** 2 / (2 * np.mean(R) ** 2))
        K, xgrid, ygrid, Kgrid = gaussian_curvature_bspline(Z, X, Y, N=100)
        fig, ax = plt.subplots(1, 2)
        color = ax[0].scatter(xgrid, ygrid, c=Kgrid, edgecolor='')
        ax[1].scatter(X, Y, c=K, edgecolor='')
        ax[0].set_title('Curvature --mesh pts')
        ax[1].set_title('Curvature --evenly spaced array pts')
        plt.colorbar(color)
        plt.show()
        print np.shape(xgrid), np.shape(ygrid), np.shape(Kgrid)
        print np.shape(X), np.shape(Y), np.shape(K)

        ## Load x,y,z from text file and compute curvature
        # fname = '/Users/npmitchell/Desktop/data_local/20151022/20151022-1120_QGP_fixbotY_Tri_N10000_dt0p000_HoR0p080_beta0p50/height.txt'
        # X,Y,Z = np.loadtxt(fname, skiprows=1, delimiter=',', unpack=True)
        # K, xgrid, ygrid, Kgrid = gaussian_curvature_unstructured(Z,X,Y,N=100)
        # fig, ax = plt.subplots(1, 2)
        # color = ax[0].scatter(xgrid,ygrid,c=Kgrid,edgecolor='')
        # ax[1].scatter(X,Y,c=K,edgecolor='')
        # plt.colorbar(color)
        # plt.show()
        # print np.shape(xgrid), np.shape(ygrid), np.shape(Kgrid)
        # print np.shape(X), np.shape(Y), np.shape(K)

    if demo_gaussiancurvature2:
        print 'Demo: Demonstrating gaussian_curvature_unstructured2: measuring the Gaussian curvature of a surface defined by a collection of points in another way.'
        x = np.random.random((5000,)).ravel() - 0.5
        y = np.random.random((5000,)).ravel() - 0.5
        sigma = 0.8
        z = sigma * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

        xy = np.dstack((x, y))[0]
        xyz = np.dstack((x, y, z))[0]
        print 'Triangulating...'
        Triang = Delaunay(xy)
        temp = Triang.vertices
        print 'Flipping orientations...'
        Tri = flip_orientations_tris(temp, xyz)
        # proxy for avg distance between points
        dist = np.mean(np.sqrt((x[Tri[:, 0]] - x[Tri[:, 1]]) ** 2 + (y[Tri[:, 0]] - y[Tri[:, 1]]) ** 2))
        dx = dist * 0.5
        print 'x=', x
        print 'y=', y

        K, xgrid, ygrid, Kgrid = gaussian_curvature_unstructured(x, y, z, dx, N=3)
        print 'shape(K)=', np.shape(K)
        fig, ax = plt.subplots(1, 2)
        color = ax[0].scatter(x, y, c=K, edgecolor='')
        ax[1].scatter(xgrid.ravel(), ygrid.ravel(), c=Kgrid.ravel(), edgecolor='')
        plt.colorbar(color)
        fig.text(0.5, 0.94, 'GCurvature using kd-tree and lookup', horizontalalignment='center')
        plt.show()

    if demo_Ztube:
        print 'Demonstrating Z_Ttube_approx: the projection of xy points to a tube with a piecewise defined shape: flat, polynomial, exponential'
        Y = np.arange(-1, 1, 0.005)
        X = np.arange(-1, 1, 0.005)  # np.zeros_like(Y)
        X, Y = np.meshgrid(X, Y)
        x = X.ravel()
        y = Y.ravel()
        decayL = 0.05
        DT = 0.15
        coldL = 0.5
        P = 10
        L = 1.0
        z = Z_Ttube_approx(x, y, decayL=decayL, DT=DT, P=P, coldL=coldL, totLen=2., minY=np.min(y), L=L,
                           polyorder='Quartic4_2xW')
        # Compute Gaussian curvature
        fdir = '/Users/npmitchell/Dropbox/Soft_Matter/PhaseField_Modeling/FEniCS/data_out/static/'
        fname = fdir + 'Ztube_GCurvature_P' + '{0:0.2f}'.format(P) \
                + '_DT' + '{0:0.2f}'.format(DT) \
                + '_decayL' + '{0:0.2f}'.format(decayL) \
                + '_coldL' + '{0:0.2f}'.format(coldL) \
                + '_L' + '{0:0.2f}'.format(L) \
                + '.png'

        K, xgrid, ygrid, Kgrid = gaussian_curvature_bspline(z, x, y, N=100)
        fig, ax = plt.subplots(1, 2)
        color = ax[0].scatter(xgrid, ygrid, c=Kgrid, edgecolor='', cmap='coolwarm', \
                              vmin=-np.max(np.abs(Kgrid)), vmax=np.max(np.abs(Kgrid)))
        ax[0].set_xlim(np.min(xgrid), np.max(xgrid))
        ax[0].set_ylim(np.min(ygrid), np.max(ygrid))
        ax[0].set_aspect('equal')
        ax[0].set_title(r'$K(x,y)$')

        ax[1].plot(y[np.abs(x) < 0.1], K[np.abs(x) < 0.1], '.')
        ax[1].set_xlim(-0.3, 0.3)
        ax[1].set_ylabel(r'$K$')
        ax[1].set_xlabel(r'$y$')
        ax[1].set_title(r'$K(y)$')
        titletext = pf_title_QGP('Ztube', P, DT, decayL, coldL, L)
        fig.text(0.5, 0.94, titletext, horizontalalignment='center')
        plt.colorbar(color)
        plt.savefig(fname)
        plt.show()

    if demo_initial_phase_multicrack:
        xy = (np.random.random((10000, 2)) - 0.5) * 2.0
        H = np.array([-0.2, 0.0, 0.3])
        Y = np.array([-0.4, 0.0, 0.4])
        beta = np.array([0.0, 0.25 * np.pi, 0.6 * np.pi])
        W = 0.1
        a = np.array([0.20, 0.05, 0.5])
        xi = 0.05
        phi = initialPhase_vec(xy, H, Y, beta, W, a, xi, fallofftype='linear')
        pf_display_scalar(xy[:, 0], xy[:, 1], phi,
                          'Phase field for some arbitrary cracks generated by initialPhase_vec()', \
                          cmap=cm.jet)

        # Use same function for single crack
        xi = 0.15
        H = 0.1;
        Y = 0.2;
        beta = 0.25 * np.pi;
        W = 0.3;
        a = 0.3
        phi = initialPhase_vec(xy, H, Y, beta, W, a, xi, fallofftype='polygauss')
        pf_display_scalar(xy[:, 0], xy[:, 1], phi,
                          'Demonstrating using same function for single crack: initialPhase_vec()',
                          cmap=cm.jet)

    if demo_GB_elastic_theory:
        alph = 0.706
        x0 = 1.0
        R = 2.35
        nu = 0.45

        U = -0.01
        P0 = GB_P_uDirichlet(alph, x0, R, U, nu)
        print 'U =', U, ' P0 = ', P0
        U = 0.0
        P0 = GB_P_uDirichlet(alph, x0, R, U, nu)
        print 'U =', U, ' P0 = ', P0
        U = 0.012
        P0 = GB_P_uDirichlet(alph, x0, R, U, nu)
        print 'U =', U, ' P0 = ', P0
        U = 0.03
        P0 = GB_P_uDirichlet(alph, x0, R, U, nu)
        print 'U =', U, ' P0 = ', P0

        alph = 0.706446
        P = 0.0679
        x0 = 0.0255319
        R = 0.06
        nu = 0.45
        U0 = GB_U_from_P(alph, x0, R, P, nu)
        print 'U0 = ', U0
        print 'GB_U_from_P(alph,x0,R,P,nu) =>>  U =  0.0153414532992', '\n'

        U0 = GB_U_from_P(alph, x0, R, 0.01, nu)
        print 'P=0.01, U0 = ', U0
        U0 = GB_U_from_P(alph, x0, R, 0.02, nu)
        print 'P=0.02, U0 = ', U0
        U0 = GB_U_from_P(alph, x0, R, 0.03, nu)
        print 'P=0.03, U0 = ', U0
        U0 = GB_U_from_P(alph, x0, R, 0.04, nu)
        print 'P=0.04, U0 = ', U0
        U0 = GB_U_from_P(alph, x0, R, 0.05, nu)
        print 'P=0.05, U0 = ', U0
        U0 = GB_U_from_P(alph, x0, R, 0.06, nu)
        print 'P=0.06, U0 = ', U0, '\n'

        alph = 0.706446
        P = 0.0679
        x0 = 0.0255319
        R = 0.12
        nu = 0.45
        U0 = GB_U_from_P(alph, x0, R, P, nu)
        print 'U0 = ', U0
        print 'GB_U_from_P(alph,x0,R,P,nu) =>>  U =  0.0153414532992', '\n'

        U0 = GB_U_from_P(alph, x0, R, 0.01, nu)
        print 'P=0.01, U0 = ', U0
        U0 = GB_U_from_P(alph, x0, R, 0.02, nu)
        print 'P=0.02, U0 = ', U0
        U0 = GB_U_from_P(alph, x0, R, 0.03, nu)
        print 'P=0.03, U0 = ', U0
        U0 = GB_U_from_P(alph, x0, R, 0.04, nu)
        print 'P=0.04, U0 = ', U0
        U0 = GB_U_from_P(alph, x0, R, 0.05, nu)
        print 'P=0.05, U0 = ', U0
        U0 = GB_U_from_P(alph, x0, R, 0.06, nu)
        print 'P=0.06, U0 = ', U0, '\n'

        alph = 0.4242640687119285
        U0 = GB_U_from_P(alph, 0.02553, 0.06, P0, nu)
        print 'U0 = ', U0

        alph = 0.2121320343559643
        U0 = GB_U_from_P(alph, 1., 2.35, P0, 0.5)
        print 'U0 = ', U0
        U0 = GB_U_from_P(0.000, 1., 2.35, P0, 0.5)
        print 'U0 = ', U0
