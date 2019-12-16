import phasefield_fluids as pe
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import copy
import lepm.lattice_elasticity as le
import lepm.line_segments as lsegs
from matplotlib.collections import LineCollection
import pickle as pkl

"""Build xml file for mesh generation in FEniCS xml mesh format"""


def write_mesh_ILPM2dolf(mesh_ilpm, fname):
    """Create a dolfin (FEniCS) mesh from ILPM mesh by saving it to a FEniCS-compatible XML file.
    The input mesh can be 2D or 3D.
    
    Parameters
    ----------
    mesh : instance of ILPM Mesh class
        a mesh created using ilpm.mesh
    fname : string
        the complete filename path for outputting as xml
        
    Returns
    ----------
    out : either string or fenics mesh
        If dolfin is available in your current python environment, returns the input mesh as dolfin mesh.
        Otherwise, saves the mesh file but returns a string statement that could not import dolfin from current environment.
    """
    pts = mesh.points
    tri = mesh.triangles

    if np.shape(pts)[1] == 2:
        ################
        # Write header #
        ################
        print('Writing header...')
        with open(fname, 'w') as myfile:
            myfile.write('<?xml version="1.0" encoding="UTF-8"?>\n\n')
        with open(fname, 'a') as myfile:
            myfile.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
            myfile.write('  <mesh celltype="triangle" dim="2">\n')
            myfile.write('    <vertices size="' + str(len(xypts)) + '">\n')

        # write vertices (positions)
        print('Writing vertices...')
        with open(fname, 'a') as myfile:
            for i in range(len(xypts)):
                myfile.write('      <vertex index="' + str(i) + '" x="' + '{0:.9E}'.format(
                    xypts[i, 0]) + '" y="' + '{0:.9E}'.format(xypts[i, 1]) + '"/>\n')

            myfile.write('    </vertices>\n')

            # write the bonds
        print('Writing triangular elements...')
        with open(fname, 'a') as myfile:
            myfile.write('    <cells size="' + str(len(TRI)) + '">\n')
            for i in range(len(TRI)):
                myfile.write('      <triangle index="' + str(i) + '" v0="' + str(TRI[i, 0]) + '" v1="' + str(
                    TRI[i, 1]) + '" v2="' + str(TRI[i, 2]) + '"/>\n')

            myfile.write('    </cells>\n')

        with open(fname, 'a') as myfile:
            myfile.write('  </mesh>\n')

        with open(fname, 'a') as myfile:
            myfile.write('</dolfin>')

        print('done! Wrote mesh to: ', fname)
        try:
            import dolfin as dolf
            print('loading meshfile = ', meshfile)
            out = dolf.Mesh(meshfile)
            return out
        except:
            return 'Wrote xml file but could not import dolfin in current python environment.'


def generate_mesh_xml_fenics(shape, N, R, LatticeTop, eta=0., theta=0., force_plot=False,
                             outdir='../meshes/', Trisel_params={}):
    """Generate a mesh in xml format, readable by FEniCS.
    The mesh has a global shape, N sites along 2R in dimension, a lattice topology, randomization eta,
    and overall rotation theta.
    Since Trisel (which selects a region in which to make dense points) doesn't work well, consider getting rid of it.
    
    Parameters
    ----------
    shape : string
        overall shape of the mesh ('square' 'rectangle2x1' 'rectangle1x2' 'circle')
    N : int
        approximate number of points in the mesh
    R : float
        length scale of the mesh (radius, half width of narrowest dimension)
    LatticeTop : string
        Lattice topology --> Triangular, SquareLatt, Vogelmethod_Disc, Trisel
        Trisel selects a region via other keywords to be denser than the rest of the mesh
    eta : float
        randomization of the mesh, in units of lattice spacing
    theta : float
        overall rotation of the mesh lattice vectors in radians
    force_plot : boolean
        whether to plot the result at the end (could be very slow to plot if True)
    Trisel_params : dict
        for Trisel only, otherwise empty. Would contain:
        Nsp : int -- would be approx number of points for sparse region, if whole sample was sparse
        H : float -- x coord for dense region
        Y : float -- for Trisel only, y coord for dense region
        beta : float -- for Trisel only, orientation of rectangular dense region
        W : float -- width of rectangular dense region
        
    Returns
    ----------
    """
    if LatticeTop == 'Trisel':
        Nsp = Trisel['Nsp'];
        H = Trisel['H'], Y = Trisel['Y'];
        beta = Trisel['beta'];
        W = Trisel['W']
    # RECTANGLE
    if shape in ['rectangle4x1', 'rectangle3x1']:
        if shape == 'rectangle3x1':
            L = np.ceil(np.sqrt(N / 3.))  # --> L x 3*L = N
        elif shape == 'rectangle4x1':
            L = np.ceil(np.sqrt(N / 4.))  # --> L x 4*L = N
        print(('L=' + str(L)))

        if LatticeTop == 'SquareLatt':
            # Establish square lattice, rotated by theta
            if abs(theta) < 1e-5:
                xypts_tmp = pe.generate_lattice([3 * L, 6 * L], [[1, 0], [0, 1]])
                add_exten = ''
            else:
                latticevecs = [[np.cos(theta), np.sin(theta)], [np.sin(theta), np.cos(theta)]]
                xypts_tmp = pe.generate_lattice([3 * L, 3 * L], latticevecs)
                add_exten = '_theta' + '{0:.3f}'.format(theta / np.pi).replace('.', 'p') + 'pi'
        elif LatticeTop == 'Triangular':
            latticevecs = [[1, 0], [0.5, np.sqrt(3) * 0.5]]
            xypts_tmp = pe.generate_lattice([3 * L, 6 * L], latticevecs)
            if theta == 0:
                add_exten = ''
            else:
                add_exten = '_theta' + '{0:.3f}'.format(theta / np.pi).replace('.', 'p') + 'pi'
                # ROTATE BY THETA
                print('Rotating by theta= ', theta, '...')
                xys = copy.deepcopy(xypts_tmp)
                xypts_tmp = np.array([[x * np.cos(theta) - y * np.sin(theta), y * np.cos(theta) + x * np.sin(theta)]
                                      for x, y in xys])
                print('max x = ', max(xypts_tmp[:, 0]))
                print('max y = ', max(xypts_tmp[:, 1]))

        # mask to rectangle
        tmp2 = xypts_tmp * 2 * R / L
        if shape == 'rectangle4x1':
            xyt = tmp2[np.logical_and(np.abs(tmp2[:, 0]) < R * 1.000000001, np.abs(tmp2[:, 1]) < (4 * R * 1.0000001)),
                  :]
        elif shape == 'rectangle3x1':
            xyt = tmp2[np.logical_and(np.abs(tmp2[:, 0]) < R * 1.000000001, np.abs(tmp2[:, 1]) < (3 * R * 1.0000001)),
                  :]
        xy = xyt / (2 * R / L)

        if LatticeTop == 'SquareLatt':
            methodexten = 'Rand'  # ('Rand' 'Right' 'Left' 'Checker')
            methodstr = 'SquareLatt' + methodexten
            if methodexten == 'Rand':
                xypts_skew = xy
            elif methodexten == 'Checker':
                xypts_skew = np.dstack((xy[:, 0] + 0.1 * np.mod(xy[:, 1], 4 * R / L), xy[:, 1]))[0]
            elif methodexten == 'Right':
                print('Skewing pts...')
                xypts_skew = np.dstack((xy[:, 0] + 0.1 * xy[:, 1], xy[:, 1]))[0]
                # print(xypts_skew)
        elif LatticeTop == 'Triangular':
            print('Triangular: skipping skew...')
            xypts_skew = xy
            methodstr = LatticeTop

        print('Triangulating points...\n')
        tri = Delaunay(xypts_skew)
        TRItmp = tri.vertices

        print('Computing bond list...\n')
        BL = pe.Tri2BL(TRItmp)
        # bL = bond_length_list(xy,BL)
        thres = np.sqrt(2.0) * 1.000001  # cut off everything longer than a diagonal
        print(('thres = ' + str(thres)))
        # dist = np.sqrt( (xy[BL[100,1],1]-xy[BL[100,0],1])**2 +(xy[BL[100,1],0]-xy[BL[100,0],0])**2 )
        # print('random bond length = '+str(dist))
        print('Trimming bond list...\n')
        BLtrim = pe.cut_bonds(BL, xy, thres)
        print(('Trimmed ' + str(len(BL) - len(BLtrim)) + ' bonds.'))

        print('Recomputing TRI...\n')
        TRI = pe.BL2TRI(BLtrim)

        # scale lattice down to size
        if eta == 0:
            xypts = xy * 2 * R / L
        else:
            print('Randomizing lattice by eta=', eta)
            jitter = eta * np.random.rand(np.shape(xy)[0], np.shape(xy)[1])
            xypts = np.dstack((xy[:, 0] + jitter[:, 0], xy[:, 1] + jitter[:, 1]))[0] * 2 * R / L

    if shape == 'rectangle2x1':
        L = np.ceil(np.sqrt(N * 0.5))  # --> L x 2*L = N
        print(('L=' + str(L)))

        if LatticeTop == 'SquareLatt':
            # Establish square lattice, rotated by theta
            if abs(theta) < 1e-5:
                xypts_tmp = pe.generate_lattice([3 * L, 3 * L], [[1, 0], [0, 1]])
                add_exten = ''
            else:
                latticevecs = [[np.cos(theta), np.sin(theta)], [np.sin(theta), np.cos(theta)]]
                xypts_tmp = pe.generate_lattice([3 * L, 3 * L], latticevecs)
                add_exten = '_theta' + '{0:.3f}'.format(theta / np.pi).replace('.', 'p') + 'pi'
        elif LatticeTop == 'Triangular':
            latticevecs = [[1, 0], [0.5, np.sqrt(3) * 0.5]]
            xypts_tmp = pe.generate_lattice([3 * L, 3 * L], latticevecs)
            if theta == 0:
                add_exten = ''
            else:
                add_exten = '_theta' + '{0:.3f}'.format(theta / np.pi).replace('.', 'p') + 'pi'
                # ROTATE BY THETA
                print('Rotating by theta= ', theta, '...')
                xys = copy.deepcopy(xypts_tmp)
                xypts_tmp = np.array([[x * np.cos(theta) - y * np.sin(theta), y * np.cos(theta) + x * np.sin(theta)]
                                      for x, y in xys])
                print('max x = ', max(xypts_tmp[:, 0]))
                print('max y = ', max(xypts_tmp[:, 1]))

        # mask to rectangle
        tmp2 = xypts_tmp * 2 * R / L
        xyt = tmp2[np.logical_and(np.abs(tmp2[:, 0]) < R * 1.000000001, np.abs(tmp2[:, 1]) < (2 * R * 1.0000001)), :]
        xy = xyt / (2 * R / L)

        if LatticeTop == 'SquareLatt':
            methodexten = 'Rand'  # ('Rand' 'Right' 'Left' 'Checker')
            methodstr = 'SquareLatt' + methodexten
            if methodexten == 'Rand':
                xypts_skew = xy
            elif methodexten == 'Checker':
                xypts_skew = np.dstack((xy[:, 0] + 0.1 * np.mod(xy[:, 1], 4 * R / L), xy[:, 1]))[0]
            elif methodexten == 'Right':
                print('Skewing pts...')
                xypts_skew = np.dstack((xy[:, 0] + 0.1 * xy[:, 1], xy[:, 1]))[0]
                # print(xypts_skew)
        elif LatticeTop == 'Triangular':
            print('Triangular: skipping skew...')
            xypts_skew = xy
            methodstr = LatticeTop

        print('Triangulating points...\n')
        tri = Delaunay(xypts_skew)
        TRItmp = tri.vertices

        print('Computing bond list...\n')
        BL = pe.Tri2BL(TRItmp)
        # bL = bond_length_list(xy,BL)
        thres = np.sqrt(2.0) * 1.000001  # cut off everything longer than a diagonal
        print(('thres = ' + str(thres)))
        # dist = np.sqrt( (xy[BL[100,1],1]-xy[BL[100,0],1])**2 +(xy[BL[100,1],0]-xy[BL[100,0],0])**2 )
        # print('random bond length = '+str(dist))
        print('Trimming bond list...\n')
        BLtrim = pe.cut_bonds(BL, xy, thres)
        print(('Trimmed ' + str(len(BL) - len(BLtrim)) + ' bonds.'))

        print('Recomputing TRI...\n')
        TRI = pe.BL2TRI(BLtrim)

        # scale lattice down to size
        if eta == 0:
            xypts = xy * 2 * R / L
        else:
            print('Randomizing lattice by eta=', eta)
            jitter = eta * np.random.rand(np.shape(xy)[0], np.shape(xy)[1])
            xypts = np.dstack((xy[:, 0] + jitter[:, 0], xy[:, 1] + jitter[:, 1]))[0] * 2 * R / L

    elif shape == 'rectangle1x2':
        L = np.ceil(np.sqrt(N * 2.0))  # --> L x 2*L = N
        print(('L=' + str(L)))

        if LatticeTop == 'SquareLatt':
            # Establish square lattice, rotated by theta
            if abs(theta) < 1e-5:
                xypts_tmp = pe.generate_lattice([3 * L, 3 * L], [[1, 0], [0, 1]])
                add_exten = ''
            else:
                latticevecs = [[np.cos(theta), np.sin(theta)], [np.sin(theta), np.cos(theta)]]
                xypts_tmp = pe.generate_lattice([3 * L, 3 * L], latticevecs)
                add_exten = '_theta' + '{0:.3f}'.format(theta / np.pi).replace('.', 'p') + 'pi'
        elif LatticeTop == 'Triangular':
            latticevecs = [[1, 0], [0.5, np.sqrt(3) * 0.5]]
            xypts_tmp = pe.generate_lattice([3 * L, 3 * L], latticevecs)
            if theta == 0:
                add_exten = ''
            else:
                add_exten = '_theta' + '{0:.3f}'.format(theta / np.pi).replace('.', 'p') + 'pi'
                # ROTATE BY THETA
                print('Rotating by theta= ', theta, '...')
                xys = copy.deepcopy(xypts_tmp)
                xypts_tmp = np.array([[x * np.cos(theta) - y * np.sin(theta), y * np.cos(theta) + x * np.sin(theta)]
                                      for x, y in xys])
                print('max x = ', max(xypts_tmp[:, 0]))
                print('max y = ', max(xypts_tmp[:, 1]))

        # mask to rectangle
        tmp2 = xypts_tmp * 2 * R / L
        xyt = tmp2[np.logical_and(np.abs(tmp2[:, 0]) < R * 2.000000001, np.abs(tmp2[:, 1]) < (R * 1.0000001)), :]
        xy = xyt / (2 * R / L)

        if LatticeTop == 'SquareLatt':
            methodexten = 'Rand'  # ('Rand' 'Right' 'Left' 'Checker')
            methodstr = 'SquareLatt' + methodexten
            if methodexten == 'Rand':
                xypts_skew = xy
            elif methodexten == 'Checker':
                xypts_skew = np.dstack((xy[:, 0] + 0.1 * np.mod(xy[:, 1], 4 * R / L), xy[:, 1]))[0]
            elif methodexten == 'Right':
                print('Skewing pts...')
                xypts_skew = np.dstack((xy[:, 0] + 0.1 * xy[:, 1], xy[:, 1]))[0]
                # print(xypts_skew)
        elif LatticeTop == 'Triangular':
            print('Triangular: skipping skew...')
            xypts_skew = xy
            methodstr = LatticeTop

        print('Triangulating points...\n')
        tri = Delaunay(xypts_skew)
        TRItmp = tri.vertices

        print('Computing bond list...\n')
        BL = pe.Tri2BL(TRItmp)
        # bL = bond_length_list(xy,BL)
        thres = np.sqrt(2.0) * 1.000001  # cut off everything longer than a diagonal
        print(('thres = ' + str(thres)))
        # dist = np.sqrt( (xy[BL[100,1],1]-xy[BL[100,0],1])**2 +(xy[BL[100,1],0]-xy[BL[100,0],0])**2 )
        # print('random bond length = '+str(dist))
        print('Trimming bond list...\n')
        BLtrim = pe.cut_bonds(BL, xy, thres)
        print(('Trimmed ' + str(len(BL) - len(BLtrim)) + ' bonds.'))

        print('Recomputing TRI...\n')
        TRI = pe.BL2TRI(BLtrim)

        # scale lattice down to size
        if eta == 0:
            xypts = xy * 2 * R / L
        else:
            print('Randomizing lattice by eta=', eta)
            jitter = eta * np.random.rand(np.shape(xy)[0], np.shape(xy)[1])
            xypts = np.dstack((xy[:, 0] + jitter[:, 0], xy[:, 1] + jitter[:, 1]))[0] * 2 * R / L

    else:
        if LatticeTop == 'Vogelmethod_Disc':
            if N == 100000:
                nBound = 900
            elif N == 80000:
                nBound = 800
            elif N == 40000:
                nBound = 500
            elif N == 10000:
                nBound = 300
            elif N == 1000:
                nBound = 100
            nVogel = N - nBound
            fraction_edge_gap = np.pi / nBound

            xypts = pe.generate_diskmesh_vogelgap(R, nVogel, nBound, fraction_edge_gap)
            # xypts += np.array([abs(np.min(xypts[:,0])) , abs(np.min(xypts[:,1])) ])

            methodstr = 'Vogel'
            add_exten = ''

            print('Triangulating points...\n')
            tri = Delaunay(xypts)
            TRI = tri.vertices

        elif LatticeTop == 'SquareLatt' or LatticeTop == 'Triangular':
            L = np.ceil(np.sqrt(4 * N / np.pi))
            print(('L=' + str(L)))

            if LatticeTop == 'SquareLatt':
                # Establish square lattice, rotated by theta
                if abs(theta) < 1e-5:
                    xypts_tmp = pe.generate_lattice([L, L], [[1, 0], [0, 1]])
                    add_exten = ''
                else:
                    latticevecs = [[np.cos(theta), np.sin(theta)], [np.sin(theta), np.cos(theta)]]
                    xypts_tmp = pe.generate_lattice([L, L], latticevecs)
                    add_exten = '_theta' + '{0:.3f}'.format(theta / np.pi).replace('.', 'p') + 'pi'
            elif LatticeTop == 'Triangular':
                latticevecs = [[1, 0], [0.5, np.sqrt(3) * 0.5]]
                xypts_tmp = pe.generate_lattice([L, L], latticevecs)
                if theta == 0:
                    add_exten = ''
                else:
                    add_exten = '_theta' + '{0:.3f}'.format(theta / np.pi).replace('.', 'p') + 'pi'
                    # ROTATE BY THETA
                    print('Rotating by theta= ', theta, '...')
                    xys = copy.deepcopy(xypts_tmp)
                    xypts_tmp = np.array([[x * np.cos(theta) - y * np.sin(theta), y * np.cos(theta) + x * np.sin(theta)]
                                          for x, y in xys])
                    print('max x = ', max(xypts_tmp[:, 0]))
                    print('max y = ', max(xypts_tmp[:, 1]))

            if shape == 'circle':
                tmp2 = xypts_tmp * 2 * R / L
                xyt = tmp2[tmp2[:, 0] ** 2 + tmp2[:, 1] ** 2 < (R * 1.00001) ** 2, :]
                xy = xyt / (2 * R / L)
            else:
                xy = xypts_tmp

            if LatticeTop == 'SquareLatt':
                methodexten = 'Rand'  # ('Rand' 'Right' 'Left' 'Checker')
                methodstr = 'SquareLatt' + methodexten
                if methodexten == 'Rand':
                    xypts_skew = xy
                elif methodexten == 'Checker':
                    xypts_skew = np.dstack((xy[:, 0] + 0.1 * np.mod(xy[:, 1], 4 * R / L), xy[:, 1]))[0]
                elif methodexten == 'Right':
                    print('Skewing pts...')
                    xypts_skew = np.dstack((xy[:, 0] + 0.1 * xy[:, 1], xy[:, 1]))[0]
                    # print(xypts_skew)
            elif LatticeTop == 'Triangular':
                print('Triangular: skipping skew...')
                xypts_skew = xy
                methodstr = LatticeTop

            print('Triangulating points...\n')
            tri = Delaunay(xypts_skew)
            TRItmp = tri.vertices

            print('Computing bond list...\n')
            BL = pe.Tri2BL(TRItmp)
            # bL = bond_length_list(xy,BL)
            thres = np.sqrt(2.0) * 1.000001  # cut off everything longer than a diagonal
            print(('thres = ' + str(thres)))
            # dist = np.sqrt( (xy[BL[100,1],1]-xy[BL[100,0],1])**2 +(xy[BL[100,1],0]-xy[BL[100,0],0])**2 )
            # print('random bond length = '+str(dist))
            print('Trimming bond list...\n')
            BLtrim = pe.cut_bonds(BL, xy, thres)
            print(('Trimmed ' + str(len(BL) - len(BLtrim)) + ' bonds.'))

            print('Recomputing TRI...\n')
            TRI = pe.BL2TRI(BLtrim)

            # scale lattice down to size
            if eta == 0:
                xypts = xy * 2 * R / L
            else:
                print('Randomizing lattice by eta=', eta)
                jitter = eta * np.random.rand(np.shape(xy)[0], np.shape(xy)[1])
                xypts = np.dstack((xy[:, 0] + jitter[:, 0], xy[:, 1] + jitter[:, 1]))[0] * 2 * R / L

        elif LatticeTop == 'Trisel':
            methodstr = LatticeTop
            add_exten = '_Nsp' + str(Nsp) + '_H' + '{0:.2f}'.format(H).replace('.', 'p') + \
                        '_Y' + '{0:.2f}'.format(Y).replace('.', 'p') + \
                        '_beta' + '{0:.2f}'.format((beta + theta) / np.pi).replace('.', 'p') + \
                        '_theta' + '{0:.2f}'.format(theta / np.pi).replace('.', 'p')

            L = np.ceil(np.sqrt(4 * N / np.pi))
            print(('L=' + str(L)))
            latticevecs = [[1, 0], [0.5, np.sqrt(3) * 0.5]]
            # Dense part of mesh
            xy_dens = pe.generate_lattice([L, L], latticevecs) * 2 * R / L

            # Sparse part of mesh
            Lsp = np.ceil(np.sqrt(4 * Nsp / np.pi))
            xy_sprs = pe.generate_lattice([Lsp, Lsp], latticevecs) * 2 * R / Lsp
            print('max dense x=', np.max(xy_dens[:, 0]))
            print('max sparse x=', np.max(xy_sprs[:, 0]))

            # Define shape of dense
            if H == 0:
                endpt1 = R * np.array([-np.cos(beta), -np.sin(beta)])
                endpt2 = R * np.array([np.cos(beta), np.sin(beta)])

            # Cut out dense part
            keep = np.where(pe.pts_are_near_lineseg(xy_dens, endpt1, endpt2, W))[0]
            keepsp = np.where(~pe.pts_are_near_lineseg(xy_sprs, endpt1, endpt2, W))[0]
            print('len(keep)=', len(keep))
            print('len(keepsp)=', len(keepsp))

            xyc_dens = xy_dens[keep, :]
            xyc_sprs = xy_sprs[keepsp, :]
            print('max dense_c x=', np.max(xyc_dens[:, 0]))
            print('max sparse_c x=', np.max(xyc_sprs[:, 0]))
            xypts_tmp = np.vstack((xyc_dens, xyc_sprs))
            print(xypts_tmp)

            if shape == 'circle':
                tmp2 = xypts_tmp
                xy = tmp2[tmp2[:, 0] ** 2 + tmp2[:, 1] ** 2 < (R * 1.00001) ** 2, :]
            else:
                xy = xypts_tmp

            print('Triangulating points...\n')
            tri = Delaunay(xy)
            TRItmp = tri.vertices

            print('Computing bond list...\n')
            BL = pe.Tri2BL(TRItmp)
            thres = np.sqrt(2.0) * 2 * R / Lsp * 1.000001  # cut off everything longer than a diagonal
            print(('thres = ' + str(thres)))
            print('Trimming bond list...\n')
            BLtrim = pe.cut_bonds(BL, xy, thres)
            print(('Trimmed ' + str(len(BL) - len(BLtrim)) + ' bonds.'))

            print('Recomputing TRI...\n')
            TRI = pe.BL2TRI(BLtrim)

            # scale lattice down to size
            if eta == 0:
                xys = xy
            else:
                print('Randomizing lattice by eta=', eta)
                interparticle_spacing = np.sqrt((xy[TRI[0, 0], 0] - xy[0, 0]) ** 2 + (xy[TRI[0, 0], 1] - xy[0, 1]) ** 2)
                jitter = eta * np.random.rand(np.shape(xy)[0], np.shape(xy)[1]) * interparticle_spacing
                xys = np.dstack((xy[:, 0] + jitter[:, 0], xy[:, 1] + jitter[:, 1]))[0]

            # ROTATE BY THETA
            xypts = np.array(
                [[x * np.cos(theta) - y * np.sin(theta), y * np.cos(theta) + x * np.sin(theta)] for x, y in xys])
            print('max x = ', max(xypts[:, 0]))
            print('max y = ', max(xypts[:, 1]))

    # Naming 
    Rstr = '{0:.3f}'.format(R).replace('.', 'p')
    etastr = '{0:.3f}'.format(eta).replace('.', 'p')
    exten = shape + 'Mesh_' + methodstr + add_exten + '_eta' + etastr + '_R' + Rstr + '_N' + str(N)
    fname = outdir + exten + '.xml'
    print('Writing to file: ', fname)

    ################
    # Write header #
    ################
    print('Writing header...')
    with open(fname, 'w') as myfile:
        myfile.write('<?xml version="1.0" encoding="UTF-8"?>\n\n')
    with open(fname, 'a') as myfile:
        myfile.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
        myfile.write('  <mesh celltype="triangle" dim="2">\n')
        myfile.write('    <vertices size="' + str(len(xypts)) + '">\n')

    # write vertices (positions)
    print('Writing vertices...')
    with open(fname, 'a') as myfile:
        for i in range(len(xypts)):
            myfile.write(
                '      <vertex index="' + str(i) + '" x="' + '{0:.9E}'.format(xypts[i, 0]) + '" y="' + '{0:.9E}'.format(
                    xypts[i, 1]) + '"/>\n')

        myfile.write('    </vertices>\n')

        # write the bonds
    print('Writing triangular elements...')
    with open(fname, 'a') as myfile:
        myfile.write('    <cells size="' + str(len(TRI)) + '">\n')
        for i in range(len(TRI)):
            myfile.write('      <triangle index="' + str(i) + '" v0="' + str(TRI[i, 0]) + '" v1="' + str(
                TRI[i, 1]) + '" v2="' + str(TRI[i, 2]) + '"/>\n')

        myfile.write('    </cells>\n')

    with open(fname, 'a') as myfile:
        myfile.write('  </mesh>\n')

    with open(fname, 'a') as myfile:
        myfile.write('</dolfin>')

    print('done!')

    # PLOT IT
    if force_plot:
        print('Plotting the result as triangulation...')
        plt.triplot(xypts[:, 0], xypts[:, 1], TRI)
        plt.axis('equal')
        plt.savefig(outdir + exten + '.png')
        plt.show()


def generate_mesh_remove_lsegs(fname, xy, rmv_lsegs, eps=1e-9, force_plot=True, thres=None, add_params=None):
    """"""
    if fname[-4:] != '.xml':
        fname += '.xml'

    print('Triangulating points...\n')
    tri = Delaunay(xy)
    TRItmp = tri.vertices

    print('Computing bond list...\n')
    BL = pe.Tri2BL(TRItmp)
    lenBL0 = len(BL)
    # bL = bond_length_list(xy,BL)
    # dist = np.sqrt( (xy[BL[100,1],1]-xy[BL[100,0],1])**2 +(xy[BL[100,1],0]-xy[BL[100,0],0])**2 )
    # print('random bond length = '+str(dist))
    if thres is not None:
        print(('thres = ' + str(thres)))
        print('Trimming bond list...\n')
        BL = pe.cut_bonds(BL, xy, thres)

    # Check it
    # for pair in BL:
    #     plt.plot(xy[pair, 0], xy[pair, 1], 'k-')

    # for poly in rmv_lsegs:
    #     print 'poly = ', poly
    lsa = lsegs.xyBL2linesegs(xy, BL)
    kill = lsegs.linesegs_intersect_linesegs(lsa, rmv_lsegs, thres=eps)
    print('kill = ', np.where(kill))
    BL = BL[~kill]

    # sys.exit()

    print(('Trimmed ' + str(lenBL0 - len(BL)) + ' bonds.'))

    print('Recomputing TRI...\n')
    TRI = pe.BL2TRI(BL)

    print('Writing to file: ', fname)

    ################
    # Write header #
    ################
    print('Writing header...')
    with open(fname, 'w') as myfile:
        myfile.write('<?xml version="1.0" encoding="UTF-8"?>\n\n')
    with open(fname, 'a') as myfile:
        myfile.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
        myfile.write('  <mesh celltype="triangle" dim="2">\n')
        myfile.write('    <vertices size="' + str(len(xy)) + '">\n')

    # write vertices (positions)
    print('Writing vertices...')
    with open(fname, 'a') as myfile:
        for i in range(len(xy)):
            myfile.write(
                '      <vertex index="' + str(i) + '" x="' + '{0:.9E}'.format(xy[i, 0]) +
                '" y="' + '{0:.9E}'.format(xy[i, 1]) + '"/>\n')

        myfile.write('    </vertices>\n')

        # write the bonds
    print('Writing triangular elements...')
    with open(fname, 'a') as myfile:
        myfile.write('    <cells size="' + str(len(TRI)) + '">\n')
        for i in range(len(TRI)):
            myfile.write('      <triangle index="' + str(i) + '" v0="' + str(TRI[i, 0]) + '" v1="' + str(
                TRI[i, 1]) + '" v2="' + str(TRI[i, 2]) + '"/>\n')

        myfile.write('    </cells>\n')

    with open(fname, 'a') as myfile:
        myfile.write('  </mesh>\n')

    with open(fname, 'a') as myfile:
        myfile.write('</dolfin>')

    ########################
    print('done! Writing pkl with linesegs and additional parameters...')
    with open(fname[0:-4] + '.pkl', "wb") as fn:
        params = {'rmv_lsegs': rmv_lsegs,
                  'xy': rmv_lsegs}
        # Add additional parameters, if supplied
        if add_params is not None:
            for key in add_params:
                params[key] = add_params[key]
                print('added ' + key + ' to params')

        pkl.dump(params, fn)

    # PLOT IT
    if force_plot:
        fnamebase = fname.split('.xml')[0]
        print('Plotting the result as triangulation...')
        # plt.triplot(xypts[:, 0], xypts[:, 1], TRI)
        bl_tmp = le.TRI2BL(TRI)
        bondslist = []
        for pair in bl_tmp:
            bondslist.append([(xy[pair[0], 0], xy[pair[0], 1]),
                              (xy[pair[1], 0], xy[pair[1], 1])])
        line_segments = LineCollection(bondslist,
                                       linewidths=1,
                                       linestyles='solid')
        ax = plt.gca()
        ax.add_collection(line_segments)
        for pair in rmv_lsegs:
            inds0, inds1 = [0, 2], [1, 3]
            plt.plot(pair[inds0], pair[inds1], 'r-')

        plt.axis('equal')
        print('saving to ' + fnamebase + '.png')
        plt.savefig(fnamebase + '.png')
        # plt.show()

        print('top = ', np.max(xy[:, 1]))

    return xy, TRI


if __name__ == '__main__':
    ##############
    # Parse Args #
    ##############
    import argparse

    parser = argparse.ArgumentParser(description='Build mesh for use with fenics.')
    parser.add_argument('-box', '--box', help='create a box with interior boundaries', action='store_true')
    parser.add_argument('-N', '--N', help='Number of points in the mesh (rough)', type=int, default=40000)
    parser.add_argument('-R', '--R', help='Radius (or half-width) of the mesh', type=float, default=1.0)
    parser.add_argument('-LT', '--LatticeTop', help='Topology of mesh (Vogelmethod_Disc SquareLatt Triangular Trisel)',
                        type=str, default='Triangular')
    parser.add_argument('-shape', '--shape', help='Shape of mesh (square circle rectangle2x1 rectangle1x2)',
                        type=str, default='square')
    parser.add_argument('-eta', '--eta', help='Jitter in lattice vertex positions, as fraction of lattice spacing',
                        type=float, default=0.0)
    parser.add_argument('-theta', '--theta', help='Additional rotation of lattice vectors, as fraction of pi',
                        type=float, default=0.0)

    # Additional options for Trisel
    parser.add_argument('-Nsp', '--Nsp', help='Number of points in the dense part of mesh (rough)', type=int,
                        default=40000)
    parser.add_argument('-H', '--Hfrac', help='X position of crack, as fraction of R', type=float, default=0.2)
    parser.add_argument('-Y', '--Yfrac', help='Y position of crack, as fraction of R', type=float, default=-1.0)
    parser.add_argument('-beta', '--betafrac', help='Inclination angle of crack, as fraction of pi', type=float,
                        default=0.5)
    parser.add_argument('-W', '--Wfrac', help='Width of dense mesh region, as fraction of R', type=float, default=0.1)
    parser.add_argument('-plot', '--force_plot', help='Whether to display resulting lattice in mpl', type=int,
                        default=0)
    parser.add_argument('-outdir', '--outdir', help='Where to store mesh', type=str,
                        default='./meshes/')
    args = parser.parse_args()

    ##############
    # Parameters #
    ##############
    rsz = args.R  # 0.12*0.5 	# meters
    nsz = args.N  # points
    LatticeTop = args.LatticeTop  # ('Vogelmethod_Disc' 'SquareLatt' 'Triangular' 'Trisel')
    shape = args.shape  # ('circle' 'square' 'rectangle2x1' 'rectangle1x2')
    eta = args.eta  # randomization (jitter)
    theta = args.theta * np.pi  # rotation of lattice vecs wrt x,y
    force_plot = args.force_plot

    #############
    # Make mesh #
    #############
    if args.box:
        # Example code for creating a box with interior boundaries
        sz = np.ceil(np.sqrt(nsz))
        latticevecs = [[1, 0], [0.5, np.sqrt(3) * 0.5]]
        xy = pe.generate_lattice([2 * sz, 2 * sz], latticevecs) * rsz / (2 * sz)
        xy -= np.array([np.min(xy[:, 0]), np.min(xy[:, 1])])
        xy /= np.max(xy[:, 0]) - np.min(xy[:, 0])
        extent = np.max(xy[:, 0]) - np.min(xy[:, 0])
        fname = './meshes/box_interior_walls_N' + str(len(xy)) + '_R{0:0.3f}'.format(extent).replace('.', 'p')
        rmv_lsegs = np.array([[0.3, 0.7, 0.9, 0.7]])
        rmv_lsegs *= rsz
        # for seg in rmv_lsegs:
        #     inds0 = [0, 2]
        #     inds1 = [1, 3]
        #     plt.plot(seg[inds0], seg[inds1], 'r-')
        # plt.title('Linesegs to cut')
        # plt.show()

        generate_mesh_remove_lsegs(fname, xy, rmv_lsegs, thres=1.2, force_plot=True)
    else:
        print(('Generating ' + shape + '-shaped mesh with ' + LatticeTop + ' lattice topology...'))
        generate_mesh_xml_fenics(shape, nsz, rsz, LatticeTop, eta, theta, force_plot, args.outdir, Trisel)
