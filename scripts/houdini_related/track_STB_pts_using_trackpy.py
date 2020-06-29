import os, sys, glob
import numpy as np
import pandas as pd
import h5py
import trackpy as tp
from tqdm import tqdm as tqdm
import tflow.velocity as vel
import argparse

def generate_single_dframe(h5dir, fn, dt=1 / 500.):
    h5paths = glob.glob(os.path.join(h5dir, '*.h5'))

    xs, ys, zs = [], [], []

    data = [xs, ys, zs]
    names2load = ['x', 'y', 'z']
    with h5py.File(h5paths[fn]) as f:
        for j, name in enumerate(names2load):
            data[j].append(f[name][...])
    data_list = list(zip(data[0][0], data[1][0], data[2][0]))

    df = pd.DataFrame(data_list)
    df.columns = ['x', 'y', 'z']
    df['frame'] = dt * fn
    #     df['t'] = dt * fn
    return df


def generate_dframes(h5dir, fns):
    hpaths = sorted(glob.glob(os.path.join(h5dir, '*.h5')))
    names2load = ['x', 'y', 'z']

    dfs = []

    for i, fn in enumerate(tqdm(fns)):
        xs, ys, zs = [], [], []
        data = [xs, ys, zs]
        with h5py.File(hpaths[fn]) as f:
            for j, name in enumerate(names2load):
                data[j].append(f[name][...])
        data_list = list(zip(data[0][0], data[1][0], data[2][0]))

        df = pd.DataFrame(data_list)
        df.columns = ['x', 'y', 'z']
        df['frame'] = fn
        dfs.append(df)
    return dfs


def generate_h5_all(h5dir = '/Users/takumi/Documents/research/turbulence/stb/tecplot_data/hdf5/l8p2mm_v200mms_f0p2hz_ftrig500hz_ext500us_10s_tr04_0575_1774_inc1',
        fns = np.arange(50, 150),
        dt = 1 / 500, savedir=None):
    if savedir is None:
        dirname = os.path.split(h5dir)[1]
        savedir = os.path.join(os.path.join(os.path.split(h5dir)[0], 'trackpy'), dirname)


    hpaths = sorted(glob.glob(os.path.join(h5dir, '*.h5')))
    nfiles = len(hpaths)


    dfs = generate_dframes(h5dir, np.arange(nfiles))

    tr = pd.concat(tp.link_df_iter(dfs, 0.5, pos_columns=['x', 'y'], memory=3) )

    # add velocity to dataFrame
    data = pd.DataFrame()
    for item in tqdm(set(tr.particle)):
        sub = tr[tr.particle == item]
        dframe = np.diff(sub.frame)
        deltat = dframe * dt
        ux = np.diff(sub.x) / deltat / 1000  # mm/frame -> mm/s -> m/s
        uy = np.diff(sub.y) / deltat / 1000  # mm/frame -> mm/s -> m/s
        uz = np.diff(sub.z) / deltat / 1000  # mm/frame -> mm/s -> m/s
        umag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)

        for x, y, z, ux_, uy_, uz_, umag_, frame in zip(sub.x[:-1], sub.y[:-1], sub.z[:-1], ux, uy, uz, umag,
                                                        sub.frame[:-1], ):
            if umag_ > 0.02:
                data = data.append([{'x': x,
                                     'y': y,
                                     'z': z,
                                     'ux': ux_,
                                     'uy': uy_,
                                     'uz': uz_,
                                     'umag': umag_,
                                     'frame': frame,
                                     'particle': item,
                                     }])

    data = data.reset_index()
    data = data.drop(labels='index', axis=1)
    print('done')

    datadict = {}
    for fn in tqdm(fns):
        keep = data['frame'] == fn
        df_fn = data[keep]

        datadict = {}
        datadict['x'] = df_fn.x.to_numpy()
        datadict['y'] = df_fn.y.to_numpy()
        datadict['z'] = df_fn.z.to_numpy()
        datadict['ux'] = df_fn.ux.to_numpy()
        datadict['uy'] = df_fn.uy.to_numpy()
        datadict['uz'] = df_fn.uz.to_numpy()
        datadict['umag'] = df_fn.umag.to_numpy()
        datadict['trackID'] = df_fn.particle.to_numpy()

        savepath = os.path.join(savedir, 'thd_umag0p02ms_frame%05d' % fn)
        vel.write_hdf5_dict(savepath, datadict, overwrite=True, verbose=False)

def generate_h5_for_n_particles(
        # h5dir = '/Users/takumi/Documents/research/turbulence/stb/tecplot_data/hdf5/l8p2mm_v200mms_f0p2hz_ftrig500hz_ext500us_10s_tr04_0575_1774_inc1',
        h5dir = '/Users/takumi/Documents/research/turbulence/stb/tecplot_data/hdf5/l8p2mm_v200mms_f0p2hz_ftrig500hz_ext500us_10s_tr04_0575_1774_inc1_all',
        fns2track =[90, 150, 220], savedir=None, fns='all',
        umag_thd=0.0,
        search_radius=1.0,
        memory=0):
    def find_domain(x, y, z):
        # Some useful lists
        top1 = [0, 200, 0, 200, -200, 0]  # top 1
        top2 = [-200, 0, 0, 200, -200, 0]  # top 2
        top3 = [-200, 0, 0, 200, 0, 200]  # top 3
        top4 = [0, 200, 0, 200, 0, 200]  # top 4
        bottom1 = [0, 200, -200, 0, -200, 0]  # bottom 1
        bottom2 = [-200, 0, -200, 0, -200, 0]  # bottom 2
        bottom3 = [-200, 0, -200, 0, 0, 200]  # bottom 3
        bottom4 = [0, 200, -200, 0, 0, 200]  # bottom 4

        domains = [top1, top2, top3, top4, bottom1, bottom2, bottom3, bottom4]
        #     grps = []

        for i, domain in enumerate(domains):
            xmin, xmax, ymin, ymax, zmin, zmax = domain
            if xmin < x and x < xmax and ymin < y and y < ymax and zmin < z and z < zmax:
                loc = i
        #             grps.append(1)
        #         else:
        #             grps.append(0)
        #     return grps
        return loc

    def gen_df_by_tracking_subset_of_par(tr, fns2track=[0, 1], umag_thd=0, dt=1 / 500.):
        """
        Extract particle data (position) of the particles existing at frame \in "fns"
        AND their velocities greater than "umag_thd".

        Then, compute the velocity of each particle.

        Returns a subset of the input DataFrame "tr", corresponding to the particles specified as above.


        """
        # Initialization
        pID2track = []

        # Get particle IDs to track
        for i, fn in enumerate(tqdm(fns2track)):
            # Create a new df which contain trajectory data at a specific frame
            subtr = tr[tr.frame == fn]
            pID2track += list(subtr.particle.to_numpy())
        # remove duplicates
        pID2track = list(dict.fromkeys(pID2track))

        # Create a df
        data = pd.DataFrame()
        for i, pID in enumerate(tqdm(pID2track, desc='iterating over particles being tracked')):
            subtr = tr[tr.particle == pID]

            # First figure out where particles came from
            for fn in sorted(fns):
                pData = subtr[subtr.frame == fn]
                if len(pData) != 0:
                    x, y, z = pData.x.to_numpy()[0], pData.y.to_numpy()[0], pData.z.to_numpy()[0]
                    loc_no = find_domain(x, y, z)
                    break

            # now get velocity for each particle whenever it was detected in tr

            dframe = np.diff(subtr.frame)
            deltat = dframe * dt
            ux = np.diff(subtr.x) / deltat / 1000  # mm/frame -> mm/s -> m/s
            uy = np.diff(subtr.y) / deltat / 1000  # mm/frame -> mm/s -> m/s
            uz = np.diff(subtr.z) / deltat / 1000  # mm/frame -> mm/s -> m/s
            umag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)

            # Now append to the "data" df

            try:
                for x, y, z, ux_, uy_, uz_, umag_, frame, start_, end_, tl_, n_missing_ in zip(subtr.x[:-1], subtr.y[:-1], subtr.z[:-1], \
                                                                ux, uy, uz, umag, subtr.frame[:-1],
                                                                subtr.start[:-1], subtr.end[:-1],
                                                                subtr.trail_length[:-1], subtr.n_missing[:-1], ):
                    if umag_ > umag_thd:
                        data = data.append([{'x': x,
                                             'y': y,
                                             'z': z,
                                             'ux': ux_,
                                             'uy': uy_,
                                             'uz': uz_,
                                             'umag': umag_,
                                             'frame': frame,
                                             'particle': pID,
                                             'loc_no': loc_no,
                                             'start': start_,
                                             'end': end_,
                                             'trail_length': tl_,
                                             'n_missing': n_missing_
                                             }])
            except:
                for x, y, z, ux_, uy_, uz_, umag_, frame in zip(subtr.x[:-1], subtr.y[:-1], subtr.z[:-1], \
                                                                ux, uy, uz, umag, subtr.frame[:-1], ):
                    if umag_ > umag_thd:
                        data = data.append([{'x': x,
                                             'y': y,
                                             'z': z,
                                             'ux': ux_,
                                             'uy': uy_,
                                             'uz': uz_,
                                             'umag': umag_,
                                             'frame': frame,
                                             'particle': pID,
                                             'loc_no': loc_no
                                             }])

        data = data.reset_index()
        data = data.drop(labels='index', axis=1)
        return data

    def save_df_in_h5_houdini_fmt(df, savedir, verbose=False, overwrite=True):
        fns = list(df.frame.to_numpy())
        fns = list(dict.fromkeys(fns))
        for fn in tqdm(fns):
            keep = df['frame'] == fn
            df_fn = df[keep]
            print('frane no, no of particle data to be saved: ', fn , len(df_fn))

            datadict = {}
            datadict['x'] = df_fn.x.to_numpy()
            datadict['y'] = df_fn.y.to_numpy()
            datadict['z'] = df_fn.z.to_numpy()
            datadict['u'] = df_fn.ux.to_numpy()
            datadict['v'] = df_fn.uy.to_numpy()
            datadict['w'] = df_fn.uz.to_numpy()
            datadict['|V|'] = df_fn.umag.to_numpy()
            datadict['trackID'] = df_fn.particle.to_numpy()
            datadict['loc_no'] = df_fn.loc_no.to_numpy()
            try:
                datadict['start'] = df_fn.start.to_numpy()
                datadict['end'] = df_fn.end.to_numpy()
                datadict['trail_length'] = df_fn.trail_length.to_numpy()
                datadict['n_missing'] = df_fn.n_missing.to_numpy()
                savepath = os.path.join(savedir, 'thd_umag0p00ms_frame%05d' % fn)
                vel.write_hdf5_dict(savepath, datadict, overwrite=overwrite, verbose=verbose)
            except:
                savepath = os.path.join(savedir, 'thd_umag0p00ms_frame%05d' % fn)
                vel.write_hdf5_dict(savepath, datadict, overwrite=overwrite, verbose=verbose)

    if savedir is None:
        pdir, dirname = os.path.split(h5dir)
        savedir = os.path.join(pdir, 'trackpy/track_pts_in_frame_____')
        savedir = os.path.join(savedir, dirname)

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    hpaths = sorted(glob.glob(os.path.join(h5dir, '*.h5')))
    nfiles = len(hpaths)
    if fns == 'all':
        fns = np.arange(nfiles)
    dfs = generate_dframes(h5dir, fns=fns)

    # Link all particle data
    tr = pd.concat(tp.link_df_iter(dfs, search_radius, pos_columns=['x', 'y', 'z'], memory=memory) )

    df2save = gen_df_by_tracking_subset_of_par(tr, fns2track=fns2track, umag_thd=umag_thd)
    print('... saving data at ', savedir)
    save_df_in_h5_houdini_fmt(df2save, savedir)

    print('... Done! Output is stored under... \n', savedir)



def main(**kwargs):
    # Brute Force (it could take forever)
    # generate_h5_all()
    generate_h5_for_n_particles(**kwargs)
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Adds a location index and compute ux, uy, uz to the existing data\n'
                                                 '... Location index: a number where particles originally camee from'
                                                 '...... assigned only for particles that appeared in the frames ones specify')

    # Locate Data
    parser.add_argument('-d', '--dir', type=str, default=None,
                        help='dir where h5 files are stored. h5 files should include AT LEAST x,y,z, trackID every frame\n'
                             'possibly x,y,z, ux,uy,uz, start, end, trail_length')
    parser.add_argument('-savedir', '--savedir', type=str, default=None,
                        help='dir where h5 files are stored. h5 files should include AT LEAST x,y,z, trackID every frame\n'
                             'possibly x,y,z, ux,uy,uz, start, end, trail_length')
    parser.add_argument('-f','--fns2track', nargs='+', help='frame numbers to specify particles to track', default=[90, 150, 220])
    parser.add_argument('-s', '--start', type=int, help='use files in the args.dir from start to end',
                        default=0)
    parser.add_argument('-e', '--end', type=int, help='use files in the args.dir from start to end',
                        default=None)
    # parser.add_argument('-comp_vel', '--comp_vel', type=bool, help='bool- If true, compute velocity for ALL particles',
    #                     default=False)
    args = parser.parse_args()

    print(args)
    nfiles = glob.glob(os.path.join(args.dir, '*.h5'))
    if args.end is None:
        args.end = nfiles
    fns = list(range(args.start, args.end))

    main(h5dir=args.dir, fns=fns, fns2track=args.fns2track, savedir=args.savedir)