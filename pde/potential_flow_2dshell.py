from dolfin import *
import numpy as np
try:
    import lepm.plotting.plotting as leplt
except:
    import lepm3.plotting.plotting as leplt
import lepm.plotting.colormaps as lecmaps
import lepm.line_segments as lsegs
import fenics_handling as pfh
import matplotlib.pyplot as plt
import lepm.data_handling as dh
import lepm.stringformat as sf
import cPickle as pkl
import glob
import lepm.dataio as dio
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
import argparse

'''Solve poisson equation for potential flow around a 2d shell.
Note: If you're solving a Poisson and don't specify any boundary terms, then it is equivalent
to specify grad(phi).n = 0 on the boundary. To set parallel velocity to zero, enforce Dirichlet BCs on the shell.


python potential_flow_2dshell.py -N 1000 -dshell 0.5 -thick 0.1 -phi 0.5
'''

parser = argparse.ArgumentParser(description='Solve for poential flow around a shell in 2d.')
parser.add_argument('-vtb', '--vtb', help='Velocity at top and bottom of the sample', type=float, default=1.0)
parser.add_argument('-dshell', '--dshell', help='Size of the shell', type=float, default=0.5)
parser.add_argument('-thole', '--theta_hole', help='Half size of the hole in radians', type=float, default=0.5)
parser.add_argument('-phi', '--phi', help='Angle at which center of hole points, in pi radians', type=float, default=0.)
parser.add_argument('-N', '--N', help='Number of points in the mesh (rough)', type=int, default=1000)
parser.add_argument('-R', '--R', help='Radius (or half-width) of the mesh', type=float, default=1.0)
parser.add_argument('-thick', '--thickness', help='Thickness of the shell', type=float, default=.10)

args = parser.parse_args()


# Parameters for the boundary condition
# determine how much of the compensating flux is coming from the top versus the bottom (in range 0 - 1).
# A value for vtopvbot of 0 means all bottom, 1 means all top
pistoncolor = lecmaps.light_gray()
eps = DOLFIN_EPS
big_eps = 0.02
thole = 0.5
vinfty = args.vtb
dshell = args.dshell
thick = args.thickness
Nmesh = args.N
thole = args.theta_hole

# Derived geometric variables
vstr = '{0:0.2f}'.format(vinfty).replace('.', 'p')
specstr = 'square_vNbcs_vinfty' + vstr + '_Nmesh' + str(Nmesh)

# class Potential(Expression):
#     def eval_cell(self, value, x, ufc_cell):
#         # For piston, enforce von Neumann and Dirichlet
#         # if ufc_cell.index > 10:
#         on_top = x[1] > (1.0 - DOLFIN_EPS - y1)
#         on_piston = abs(x[0] - 0.5) < dpiston and on_top
#         if on_piston:
#             value[0] = 0.0


# Create mesh and define function space
# mesh = UnitSquareMesh(100, 100)
meshspec = '_dshell' + sf.float2pstr(args.dshell) + '_thetahole' + sf.float2pstr(thole) + \
           '_phihole' + sf.float2pstr(args.phi) + '_thick' + sf.float2pstr(args.thickness)
meshfile = './meshes/shell_N' + str(Nmesh) + '_n*_R1p000' + meshspec + '.xml'
print 'searching for ', meshfile
meshfile = glob.glob(meshfile)[0]
# Load the information about the linesegments that defined the boundary, rmv_lsegs
meshparamfn = meshfile[0:-3] + 'pkl'
with open(meshparamfn, 'rb') as fn:
    pdict = pkl.load(fn)
rmv_lsegs = pdict['rmv_lsegs']

mesh = Mesh(meshfile)
Vf = FunctionSpace(mesh, "Lagrange", 1)
Vv = VectorFunctionSpace(mesh, "Lagrange", 1)
Rf = FunctionSpace(mesh, "R", 0)
# Wf = Vf + Rf
try:
    Wf = Vf * Rf
except:
    f_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
    r_ele = FiniteElement("R", mesh.ufl_cell(), 0)
    Wf = FunctionSpace(mesh, MixedElement([f_ele, r_ele]))

# Get coordinates
xy = pfh.xy_from_function_space(None, None, mesh)
maxx = np.max(xy[:, 0])
maxy = np.max(xy[:, 1])

assigner = FunctionAssigner(Vf, Wf.sub(0))
phi0 = Function(Vf)
phik0 = Constant(1.0)
phi_k = interpolate(phik0, Vf)

#######################################
# Define surfaces for von Neumann BCs
#######################################


class OnShell(SubDomain):
    def inside(self, x, on_boundary):
        in_left = x[0] > eps
        in_right = x[0] < maxx - eps
        below = x[1] < maxy - eps
        above = x[1] > eps
        return below and above and in_right and in_left and on_boundary


class OnSurface(SubDomain):
    def inside(self, x, on_boundary):
        # the boundary is everything but the side holes and top piston
        on_top = near(x[1], maxy, eps)
        return on_top and on_boundary


class OnOuterWall(SubDomain):
    def inside(self, x, on_boundary):
        # the boundary is everything but the side holes and top piston
        on_bottom = x[1] < DOLFIN_EPS
        on_side = x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS
        return (on_bottom or on_side) and on_boundary


class OnBottom(SubDomain):
    def inside(self, x, on_boundary):
        on_bottom = x[1] < DOLFIN_EPS
        return on_bottom and on_boundary


# Initialize mesh function for interior domains
domains = CellFunction("size_t", mesh)
domains.set_all(0)
# box = InBox()
# box.mark(domains, 1)

# define interface and facet measure
# facet_domains = FacetFunction("size_t", mesh)
# interface = CompiledSubDomain("fabs(x[0]-0.5)<DOLFIN_EPS")
# interface.mark(facet_domains, 1)

# Initialize mesh function for boundary domains
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
shell = OnShell()
surface = OnSurface()
bottom = OnBottom()
shell.mark(boundaries, 1)
surface.mark(boundaries, 2)
bottom.mark(boundaries, 3)
ds = Measure("ds")[boundaries]
phi0shell = Constant(0.)
# dS = Measure("dS", domain=mesh, subdomain_data=boundaries)
# dSs = dS[boundaries]

# Define variational problem
(phi, cc) = TrialFunction(Wf)
(v, dd) = TestFunctions(Wf)
vel = Constant(str(vinfty))
f = Constant(0.)
# f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
a = inner(nabla_grad(phi), nabla_grad(v)) * dx + cc * v * dx + phi * dd * dx
# L = f * v * dx + vel("+") * v("+") * dSs(1) - vel("+") * v("+") * dSs(2)

L = f * v * dx - vel * v * ds(2) + vel * v * ds(3)

# Define boundary conditions
# phi0top = Potential()
bcs = DirichletBC(Vf, phi0shell, shell)

# Compute solution
ww = Function(Wf)
solve(a == L, ww,
      solver_parameters=dict(linear_solver="cg",
                             preconditioner="ilu"), bcs=bcs)
(phi_out, cc) = ww.split()

assigner.assign(phi_k, phi_out)

# Look at value of cc
assigner_R = FunctionAssigner(Rf, Wf.sub(1))
c0 = Function(Rf)
ck0 = Constant(1.0)
c_k = interpolate(ck0, Rf)
assigner_R.assign(c_k, cc)
print 'c_k = ', c_k.vector().array()

# Hold plot
plot(mesh)
plot(phi_out)
interactive()

#############################################
# Define goal functional (quantity of interest)
# M = u * dx()
#
# # Define error tolerance
# tol = 1.e-5
#
# # Solve equation a = L with respect to u and the given boundary
# # conditions, such that the estimated error (measured in M) is less
# # than tol
# problem = LinearVariationalProblem(a, L, u, bc)
# solver = AdaptiveLinearVariationalSolver(problem, M)
# solver.parameters["error_control"]["dual_variational_solver"]["linear_solver"] = "cg"
# solver.solve(tol)
# solver.summary()
#
# # Plot solution(s)
# plot(u.root_node(), title="Solution on initial mesh")
# plot(u.leaf_node(), title="Solution on final mesh")
# interactive()

#############################################
# Look at results
#############################################
# Evaluate integral of normal gradient over top boundary
n = FacetNormal(mesh)
m1 = dot(grad(phi_k), n)*ds(2)
v1 = assemble(m1)
print("\int grad(u) * n ds(2) = ", v1)

# Evaluate integral of u over the obstacle
m2 = phi_k * dx(1)
v2 = assemble(m2)
print("\int u dx(1) = ", v2)

outdir = './results/' + 'potential_flow_shell_' + specstr + '/'
dio.ensure_dir(outdir)
# Save as png
phiv = phi_k.vector().array()
# Get xy from mesh
xy = pfh.xy_from_function_space(None, None, mesh)

# Get velocity from gradients
uu = project(grad(phi_k), Vv)
uuv = uu.vector().array().reshape(np.shape(xy))

##############################################
# Save Phi data
#############################################
with open(outdir + 'data.pkl', "wb") as fn:
    res = {'xy': xy, 'phi': phiv, 'uu': uuv}
    pkl.dump(res, fn)

