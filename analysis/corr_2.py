import numpy as np
import turbulence.analysis.corr as corr
import turbulence.analysis.statP as statP
import turbulence.manager.access as access
import turbulence.display.graphes as graphes
import matplotlib.pylab as plt
import dpath.util


def Time(M, field, tlist, N=50, norm_t=1.):
    """
    Compute the auto correlation function in time of a given field on a series of time
    Store the result in a dict format

    Parameters
    ----------
    M : Mdata class instance, or any other object that contains the following fields :
        methods : shape()
        attributes : Ux, Uy
    field :
    tlist :
    N : int
    norm :

    Returns
    -------
    ccCorr_t :

    """
    Corr_t = {}
    for i in tlist:
        t, C = corr.corr_v_t([M], i, axes=[field, field], N=N, p=1, display=False)
        Corr_t[(i, 't_' + field)] = np.asarray(t) * 1. / norm_t
        Corr_t[(i, 'C_' + field)] = C

    return Corr_t


def average(dict_corr, field, key='t'):
    sub_t = dpath.util.search(dict_corr, '*' + key + '_' + field + '*')
    sub_Ct = dpath.util.search(dict_corr, '*C_' + field + '*')

    t_moy = np.median(np.stack(sub_t.values()), axis=0)
    C_moy = np.median(np.stack(sub_Ct.values()), axis=0)

    dict_corr[('moy', key + '_' + field)] = t_moy
    dict_corr[('moy', 'C_' + field)] = C_moy

    return dict_corr


def average_global(list_corr, field, key='t', fignum=1):
    # mean is a linear operator. average first on each realization then on the realizations

    X = []
    C = []
    for dict_corr in list_corr:
        dict_corr = average(dict_corr, field, key=key)

        X.append(dict_corr[('moy', key + '_' + field)])
        C.append(dict_corr[('moy', 'C_' + field)])

    Cmoy = np.nanmean(np.asarray(C), axis=0)
    X = np.nanmean(np.asarray(X), axis=0)

    graphes.graph(X, Cmoy, label='r-', fignum=fignum)
    n = len(list_corr)
    graphes.plt.title('Ensemble average over ' + str(n) + ' realizations')


#    return X,Cmoy
# plt.plot(X[0],Cmoy)

def Space(M, field, tlist, N=30, Np=10 ** 4, norm_d=1.):
    dlist = range(N)
    dx = np.diff(M.x[0, :])[0]

    indices = {}
    Corr_d = {}

    U = access.get(M, field, 0)

    for d in dlist:
        indices[d] = corr.d_2pts_rand(U[..., 0], d, Np)

    for i in tlist:
        C = np.zeros(len(dlist))
        for d in dlist:
            C[d] = compute(M, i, indices[d], axes=[field, field])

        Corr_d[(i, 'd_' + field)] = np.asarray(dlist) * dx / norm_d
        Corr_d[(i, 'C_' + field)] = C

    return Corr_d


def compute(M, t, indices, avg=1, axes=['U', 'U'], p=1, average=False):
    C = []
    C_norm = []

    X, Y = chose_axe(M, t, axes)

    if average:
        Xmoy, Xstd = statP.average(X)
        Ymoy, Ystd = statP.average(Y)
    else:
        Xmoy = 0
        Xstd = 0
        Ymoy = 0
        Ystd = 0

    for i, j in indices.keys():
        # print(indices[i,j])
        k, l = indices[i, j]
        vec_t = [k - i, l - j]

        # Xl = project(X[i,j,...],vec_t,'l')
        # Yl = project(Y[k,l,...],vec_t,'l')

        # Xt = project(X[i,j,...],vec_t,'t')
        # Yt = project(Y[k,l,...],vec_t,'t')


        Sp = (X[i, j, ...] - Xmoy) ** p * (Y[
                                               k, l, ...] - Ymoy) ** p  # remove the average in space ?   -> remove by default
        C.append(Sp)  # for k,l in indices[(i,j)]])

        Sp_norm = ((X[i, j, ...] + Y[k, l, ...] - Xmoy - Ymoy) / 2) ** (2 * p)
        C_norm.append(Sp_norm)
        # substract the mean flow ? it shouldn't change the result so much, as there is no strong mean flow
        # -> to be checked
        # how to compute the mean flow : at which scale ? box size ? local average ?
        # Cmoy,Cstd=average(C)
    Cf = statP.average(C)[0] / statP.average(C_norm)[0]
    return Cf


def project(U, t, typ):
    vt = t / norm(t)

    if typ == 'l':
        pass
    if typ == 't':
        pass


def chose_axe(M, t, axes, Dt=1):
    """
    Chose N axis of a Mdata set
    INPUT
    -----
    M : Madata object
    t : int
        time index
    axes : string list 
        Possible values are : 'E', 'Ux', 'Uy', 'strain', 'omega'
    OUTPUT
    ----- 
    """
    data = tuple([access.get(M, ax, t, Dt=Dt) for ax in axes])
    return data

    ############# plotting #################


def summary_time(Corr_fun, tlist, field, avg=False, fignum=1):
    figs = {}

    plt.figure(fignum)
    for i in tlist:
        plt.plot(Corr_fun[(i, 't_' + field)], Corr_fun[(i, 'C_' + field)], label='ko-')
    if avg:
        plt.plot(Corr_fun[('moy', 't_' + field)], Corr_fun[('moy', 'C_' + field)], label='r-')
    #plt.axis([-0.1, 0.1, 0, 1])

    if field == 'omega':
        field = '\omega'
    figs.update(graphes.legende('Time (s)', '$Ct_{' + field + field + '}$', 'Time average over one realization'))
    return figs


def summary_space(Corr_fun, tlist, field, avg=False, fignum=1):
    figs = {}

    plt.figure(fignum)
    for i in tlist:
        plt.plot(Corr_fun[(i, 'd_' + field)], Corr_fun[(i, 'C_' + field)], 'ko-')
    if avg:
        plt.plot(Corr_fun[('moy', 'd_' + field)], Corr_fun[('moy', 'C_' + field)], 'r-')
    plt.axis([0, 12., 0, 1])

    if field == 'omega':
        field = '\omega'
    figs.update(graphes.legende('Distance (mm)', '$Cd_{' + field + field + '}$', 'Time average over one realization'))
    return figs
