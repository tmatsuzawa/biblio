from dolfin import *
import numpy as np
import lepm.plotting.colormaps as lecmaps
import fenics_handling as pfh
import lepm.stringformat as sf
import cPickle as pkl
import glob
import lepm.dataio as dio
import argparse

'''This is the principal code to run simulations of Stokes flow for all geometries.
See argparse options below.
'''

parser = argparse.ArgumentParser(description='Solve for poential flow around a shell in 2d.')
parser.add_argument('-dshell', '--dshell', help='Size of the shell', type=float, default=0.25)
parser.add_argument('-geometry', '--geometry', help='Shell or dolphin, etc', type=str, default='shell')
parser.add_argument('-thole', '--theta_hole', help='Half size of the hole in radians', type=float, default=0.5)
parser.add_argument('-phi', '--phi', help='Angle at which center of hole points, in pi radians', type=float, default=0.)
parser.add_argument('-N', '--N', help='Number of points in the mesh (rough)', type=int, default=10000)
parser.add_argument('-R', '--R', help='Radius (or half-width) of the mesh', type=float, default=1.0)
parser.add_argument('-thick', '--thickness', help='Thickness of the shell', type=float, default=.02)
parser.add_argument('-lr', '--lrbcs', help='Right and left flow boundary conditions', action='store_true')
parser.add_argument('-tb', '--tbbcs', help='Top and bottom no slip boundary conditions', action='store_true')
parser.add_argument('-uniform', '--uniform_inflow', help='Inflow on right is independent of y', action='store_true')
parser.add_argument('-plot', '--plot', help='Plot the results before saving', action='store_true')
args = parser.parse_args()

##############################
# Load mesh and subdomains   #
##############################
if args.geometry == 'dolphin':
    mesh = Mesh("./meshes/dolfin_fine.xml.gz")
    # sub_domains = MeshFunction("size_t", mesh, "./meshes/dolfin_fine_subdomains.xml.gz")
    # Parameters for the boundary condition
    pistoncolor = lecmaps.light_gray()
    eps = DOLFIN_EPS
    big_eps = 0.02
    thole = 0.5
    dshell = args.dshell
    thick = args.thickness
    Nmesh = args.N
    thole = args.theta_hole
    # Derived geometric variables
    specstr = 'square_dolphin'
    if not args.uniform_inflow:
        specstr += '_sinusoidal_inflow'

    # Create mesh and define function space
    # mesh = UnitSquareMesh(100, 100)
    meshspec = 'dolphin'
else:
    # mesh = Mesh("./meshes/dolfin_fine.xml.gz")
    # sub_domains = MeshFunction("size_t", mesh, "./meshes/dolfin_fine_subdomains.xml.gz")
    # Parameters for the boundary condition
    pistoncolor = lecmaps.light_gray()
    eps = DOLFIN_EPS
    big_eps = 0.02
    thole = 0.5
    dshell = args.dshell
    thick = args.thickness
    Nmesh = args.N
    thole = args.theta_hole
    # Derived geometric variables
    specstr = 'square_Nmesh' + str(Nmesh)
    if not args.uniform_inflow:
        specstr += '_sinusoidal_inflow'
    # Create mesh and define function space
    # mesh = UnitSquareMesh(100, 100)
    meshspec = 'dshell' + sf.float2pstr(args.dshell) + '_thetahole' + sf.float2pstr(thole) + \
               '_phihole' + sf.float2pstr(args.phi) + '_thick' + sf.float2pstr(args.thickness)
    meshfile = './meshes/shell_N' + str(Nmesh) + '_n*_R1p000_' + meshspec + '.xml'
    print 'searching for ', meshfile
    meshfile = glob.glob(meshfile)[0]
    mesh = Mesh(meshfile)

# Get coordinates
xy = pfh.xy_from_function_space(None, None, mesh)
maxx = np.max(xy[:, 0])
maxy = np.max(xy[:, 1])


class OnShell(SubDomain):
    def inside(self, x, on_boundary):
        in_left = x[0] > eps
        in_right = x[0] < maxx - eps
        below = x[1] < maxy - eps
        above = x[1] > eps
        return below and above and in_right and in_left and on_boundary


class OnTop(SubDomain):
    def inside(self, x, on_boundary):
        # the boundary is everything but the side holes and top piston
        on_top = near(x[1], maxy, eps)
        return on_top and on_boundary


class OnBottom(SubDomain):
    def inside(self, x, on_boundary):
        # the boundary is everything but the side holes and top piston
        on_bot = near(x[1], 0., eps)
        return on_bot and on_boundary


class OnRight(SubDomain):
    def inside(self, x, on_boundary):
        # the boundary is everything but the side holes and top piston
        on_right = near(x[0], maxx, eps)
        return on_right and on_boundary


class OnLeft(SubDomain):
    def inside(self, x, on_boundary):
        # the boundary is everything but the side holes and top piston
        on_left = near(x[0], 0., eps)
        return on_left and on_boundary


domains = CellFunction("size_t", mesh)
domains.set_all(0)

boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
shell = OnShell()
right = OnRight()
left = OnLeft()
top = OnTop()
bottom = OnBottom()
shell.mark(boundaries, 1)
right.mark(boundaries, 2)
left.mark(boundaries, 3)
top.mark(boundaries, 4)
bottom.mark(boundaries, 5)
ds = Measure("ds")[boundaries]

# Define function spaces
scalar = FunctionSpace(mesh, "CG", 1)
vector = VectorFunctionSpace(mesh, "CG", 1)
system = vector * scalar

# Create functions for boundary conditions
noslip = Constant((0, 0))
zero = Constant(0)
if args.uniform_inflow:
    inflow = Expression(("-1", "0"))
else:
    inflow = Expression(("-sin(x[1] * pi)", "0"))

# No-slip boundary condition for velocity
bc0 = DirichletBC(system.sub(0), noslip, shell)  # sub_domains, 1)

# Inflow boundary condition for velocity
bc1 = DirichletBC(system.sub(0), inflow, right)  # sub_domains, 2)
bc2 = DirichletBC(system.sub(0), inflow, left)  # sub_domains, 2)
bc3 = DirichletBC(system.sub(0), noslip, top)  # sub_domains, 2)
bc4 = DirichletBC(system.sub(0), noslip, bottom)  # sub_domains, 2)

# Collect boundary conditions
if args.lrbcs:
    if args.tbbcs:
        bcs = [bc0, bc1, bc2, bc3, bc4]
    else:
        bcs = [bc0, bc1, bc2]
else:
    if args.tbbcs:
        bcs = [bc0, bc1, bc3, bc4]
    else:
        bcs = [bc0, bc1]

# Define variational problem
(v, q) = TestFunctions(system)
(u, p) = TrialFunctions(system)
f = Constant((0, 0))
h = CellSize(mesh)
beta  = 0.2
delta = beta*h*h
a = (inner(grad(v), grad(u)) - div(v)*p + q*div(u) + \
    delta * inner(grad(q), grad(p)))*dx
L = inner(v + delta*grad(q), f)*dx

# Compute solution
w = Function(system)
solve(a == L, w, bcs)
uu, pp = w.split()

# Save solution in VTK format
# ufile_pvd = File("velocity.pvd")
# ufile_pvd << uu
# pfile_pvd = File("pressure.pvd")
# pfile_pvd << pp

# Plot solution
if args.plot:
    plot(uu)
    plot(pp)
    interactive()

#############################################
# Look at results
#############################################
# Assign the velocity, uu
uuk0 = Constant((1.0, 1.0))
ppk0 = Constant(1.0)
uuk = interpolate(uuk0, vector)
assigner = FunctionAssigner(vector, system.sub(0))
assigner.assign(uuk, uu)

# Assign the pressure, pp
ppk = interpolate(ppk0, scalar)
assigner_p = FunctionAssigner(scalar, system.sub(1))
assigner_p.assign(ppk, pp)

# Evaluate integral of normal gradient over shell boundary
n = FacetNormal(mesh)
m1 = dot(uuk, n) * ds(1)
v1 = assemble(m1)
print("\int u.n ds(1) = ", v1)

# Evaluate integral of normal gradient over right boundary
n = FacetNormal(mesh)
m1 = dot(uuk, n) * ds(2)
v1 = assemble(m1)
print("\int u.n ds(2) = ", v1)

if args.lrbcs:
    if args.tbbcs:
        outdir = './results/' + 'stokes_flow_shell_' + specstr + '_nosliptb/' + meshspec + '/'
    else:
        outdir = './results/' + 'stokes_flow_shell_' + specstr + '/' + meshspec + '/'
else:
    if args.tbbcs:
        outdir = './results/' + 'stokes_flow_shell_' + specstr + '_nosliptb_rightflow/' + meshspec + '/'
    else:
        outdir = './results/' + 'stokes_flow_shell_' + specstr + '_rightflow/' + meshspec + '/'

dio.ensure_dir(outdir)

# Get xy from mesh
xy = pfh.xy_from_function_space(None, None, mesh)
uuv = uuk.vector().array().reshape(np.shape(xy))
ppv = ppk.vector().array()
print 'np.shape(uuv) = ', np.shape(uuv)
print 'np.shape(uuv) = ', np.shape(ppv)

# Get boundary points
bmesh = BoundaryMesh(mesh, "exterior", True)
boundary = bmesh.coordinates()
# Split the boundary into inner and outer
b_inner0 = np.where(np.logical_and(boundary[:, 0] > big_eps, boundary[:, 0] < 1 - big_eps))[0]
b_inner1 = np.where(np.logical_and(boundary[:, 1] > big_eps, boundary[:, 1] < 1 - big_eps))[0]
b_inner = np.intersect1d(b_inner0, b_inner1)
b_outer = np.setdiff1d(np.arange(len(boundary), dtype=int), b_inner)
# Order the outer boundary by polar angle
btheta = np.arctan2(boundary[b_outer, 1], boundary[b_outer, 0])
bouter_inds = np.argsort(btheta)
bo_xy = boundary[b_outer, :]
# Also grab the inner boundary points, unordered
bi_xy = boundary[b_inner]
# # Order the inside boundary by using lattice methods
# xyb, NL, KL, BL, BM = networks.delaunay_lattice_from_pts(bi_xy, trimbound=True, max_bond_length=0.2)
# bi_inds = networks.extract_boundary(xyb, NL, KL, BL)
# bi_xy = xyb[bi_inds]

##############################################
# Save data
#############################################
outfn = outdir + 'data.pkl'
print 'outputting to fn:' + outfn
with open(outfn, "wb") as fn:
    res = {'xy': xy, 'uu': uuv, 'pp': ppv, 'boundaries': boundary, 'b_inner': bi_xy, 'b_outer': bo_xy}
    pkl.dump(res, fn)
