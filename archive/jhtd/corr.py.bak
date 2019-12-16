import turbulence.display.graphes as graphes
import numpy as np
import sys

'''
'''


def spatial_corr(data, N=1024, Dt=10, outdir='/Users/npmitchell/Dropbox/Soft_Matter/turbulence/jhtd/'):
    """Compute spatial correlation

    Parameters
    ----------
    data
    N
    Dt

    Returns
    -------

    """
    Cxx = np.zeros((N / 2, Dt))
    d = np.arange(N / 2)
    figs = {}

    for p in range(3):
        for k in range(Dt):
            key = data.keys()[k]
            Z = np.asarray(data[key])

            Ex = np.nanmean(np.power(Z[:N / 2, 0, 0, p], 2))
            Cxx[:, k] = np.nanmean(
                np.asarray([[Z[i, 0, 0, p] * Z[i + j, 0, 0, p] / Ex for i in range(N / 2)] for j in range(N / 2)]),
                axis=1)
            # print(Cxx[0,:])
        C = np.nanmean(Cxx, axis=1)
        graphes.graph(d, C, fignum=1)
        graphes.set_axis(0, N / 4, -1, 1.5)
        figs.update(graphes.legende('d', 'C', ''))

    graphes.save_figs(figs, savedir=outdir + 'corr_functions/', suffix='', prefix='', frmt='pdf', dpi=300)


def corr_fun():
    print 'jhtd.corr.corr_fun(): not implemented'
    sys.exit()
    pass


def time_corr(data, param, N=800, outdir='/Users/npmitchell/Dropbox/Soft_Matter/turbulence/jhtd/'):
    """

    Parameters
    ----------
    data
    param
    N

    Returns
    -------

    """
    keys = data.keys()

    figs = {}

    for p in range(3):
        print(p)
        keys = np.sort(keys)

        [data[key] for key in keys]
        t = np.asarray([param[key]['t0'] for key in keys[0:N / 2]])

        C = np.zeros(N / 2)
        Dt = np.zeros(N / 2)
        for i in range(N / 2):
            if i % 100 == 0:
                print(i)
            C[i] = np.nanmean(
                np.asarray([data[key1][..., p] * data[keys[i + j]][..., p] for j, key1 in enumerate(keys[0:N / 2])]),
                axis=(0, 1, 2))
            Dt[i] = np.nanmean(np.asarray([param[keys[j + i]]['t0'] - param[keys[j]]['t0'] for j in range(N / 2)]))

            #  print(Dt)
        graphes.graph(Dt, C / C[0], fignum=4)
        graphes.set_axis(0, 400, -1, 1.1)
        figs.update(graphes.legende('t', 'C', 'JHTD corr functions'))

    graphes.save_figs(figs, savedir=outdir + 'corr_functions/', suffix='', prefix='', frmt='pdf', dpi=300)

#    Cxx[:,k] = np.nanmean(np.asarray([[[i,0,0,p]*Z[i+j,0,0,p]/Ex for i in range(N/2)] for j in range(N/2)]),axis=1)
