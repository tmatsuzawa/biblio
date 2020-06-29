import os, sys, glob, time
import numpy as np
import pandas as pd
import h5py
import trackpy as tp
from tqdm import tqdm as tqdm
import random
# import tflow.velocity as vel
import argparse
import pickle

def convert_float_to_decimalstr(number, zpad=3):
    """
    Convert float into decimal string which replaces a decimal point with a letter p.
    e.g. 5.3 -> "5p3"

    Parameters
    ----------
    number: decimal str

    Returns
    -------
    str: string
    """
    number_str = str(number)
    number_str = number_str.replace('.', 'p')
    return number_str.rjust(zpad, '0')


def generate_single_dframe(h5dir, fn, trackpy_format=True, mappings_tr_stats=None):
    """
    Creates a dataFrame from a single h5 file in a directory
    ... data in h5 must have the same length
    """
    df = pd.DataFrame()

    h5paths = sorted(glob.glob(os.path.join(h5dir, '*.h5')))
    with h5py.File(h5paths[fn]) as f:
        names2load = list(f.keys())
        for j, name in enumerate(names2load):
            df[name] = f[name][...]

    # if mappings_tr_stats is not None and nmin != 0:
    #     map_pID2tls = mappings_tr_stats[0]
        # try:
        #     pIDs = df['trackID'].to_numpy()
        # except:
        #     pIDs = df['particle'][...].to_numpy()
        # keep = np.asarray([map_pID2tls[pID] > nmin for pID in pIDs])
        # df = df[keep]

    if trackpy_format:
        df['frame'] = fn

        try:
            df = df.rename(columns={'trackID': 'particle'})
            df = df.rename(columns={'u': 'ux', 'v': 'uy', 'w': 'uz', '|V|': 'umag'})
        except:
            pass

    return df


def generate_dframes(h5dir, fns='all', load_id=False, load_vel=False):
    h5paths = glob.glob(os.path.join(h5dir, '*.h5'))
    names2load = ['x', 'y', 'z']
    cnames = ['x', 'y', 'z']
    if load_vel:
        names2load += ['u', 'v', 'w']
        cnames += ['ux', 'uy', 'uz']

    dfs = []

    if fns == 'all':
        fns = np.arange(len(h5paths))

    for i, fn in enumerate(tqdm(fns)):
        xs, ys, zs = [], [], []
        data = [xs, ys, zs]
        if load_vel:
            us, vs, ws = [], [], []
            data += [us, vs, ws]

        with h5py.File(h5paths[fn]) as f:
            for j, name in enumerate(names2load):
                data[j].append(f[name][...])
            if load_id:
                trackIDs = f['trackID'][...].astype(int)

        # data_list = list(zip(data[0][0], data[1][0], data[2][0]))
        #
        # df = pd.DataFrame(data_list)
        # df.columns = ['x', 'y', 'z']
        df = pd.DataFrame()
        for j, cname in enumerate(cnames):
            df[cname] = data[j][0]

        df['frame'] = fn

        if load_id:
            df['particle'] = trackIDs
        dfs.append(df)
    return dfs


# def generate_dframes(h5dir, fns='all', dt=1 / 500.):
#     h5paths = glob.glob(os.path.join(h5dir, '*.h5'))
#     names2load = ['x', 'y', 'z']
#
#     dfs = []
#
#     if fns == 'all':
#         fns = np.arange(len(h5paths))
#
#     for i, fn in enumerate(fns):
#         xs, ys, zs = [], [], []
#         data = [xs, ys, zs]
#         with h5py.File(h5paths[fn]) as f:
#             for j, name in enumerate(names2load):
#                 data[j].append(f[name][...])
#         data_list = list(zip(data[0][0], data[1][0], data[2][0]))
#
#         df = pd.DataFrame(data_list)
#         df.columns = ['x', 'y', 'z']
#         df['frame'] = fn
#
#         dfs.append(df)
#     return dfs

def get_trail_stats_in_lists(tr):
    """
    Returns the lists of trail length, frames that each particle appeared and disappeared,
    number of particles that should have been observed
    ... How to use:
        tl[particle ID(pID)]: trail length of the particle with "pID"
    ... If one wishes to add a column about the trail length to the original trajectory dataframe,
        try following.

        tls_all = [tls[i] for i in tr['particle'].to_numpy()]
        tr['trail_length'] = tls_all

    ... sample usage:
        tr['trail_length'] =  [tls[i] for i in tr['particle'].to_numpy()]
        tr['start'] = [starts[i] for i in tr['particle'].to_numpy()]
        tr['end'] = [ends[i] for i in tr['particle'].to_numpy()]
        tr['n_missing'] = [n_missing[i] for i in tr['particle'].to_numpy()]


    """
    # add the frame when the particle was first observed
    pIDs = np.asarray(list(set(tr.particle.to_numpy())))

    starts, ends, tls = [], [], []
    n_missing = []  # number of frames that the particle should have been found
    for i, pID in enumerate(tqdm(pIDs, desc='Extracting tral statistics')):
        subtr = tr[tr.particle == pID]
        frames = subtr.frame.to_numpy()
        start = np.min(frames)
        end = np.max(frames)
        tl = end - start
        starts.append(start)
        ends.append(end)
        tls.append(tl)
        n_missing.append(end - start + 1 - len(subtr))

    return tls, starts, ends, n_missing


def get_trail_stats_in_dict(tr):
    """
    Returns the dictionaries of trail length, frames that each particle appeared and disappeared,
    number of particles that should have been observed
    ... this could be used to map a column to create aonther column in a dataframe
    """
    # add the frame when the particle was first observed
    pIDs = np.asarray(list(set(tr.particle.to_numpy())))

    starts, ends, tls = {}, {}, {}
    n_missing = {}  # number of frames that the particle should have been found
    arc_lengths = {}  # arclength of individual trajectories
    for i, pID in enumerate(tqdm(pIDs, desc='Extracting trail statistics for each particle')):
        subtr = tr[tr.particle == pID]
        frames = subtr.frame.to_numpy()
        start = np.min(frames)
        end = np.max(frames)
        tl = end - start

        starts[pID] = start
        ends[pID] = end
        tls[pID] = tl
        n_missing[pID] = end - start + 1 - len(subtr)

        # Arclength of a trajectory
        if tl == 0:
            arc_lengths[pID] = 0.
        else:
            im = tp.imsd(subtr, 1., 1., max_lagtime=np.inf, pos_columns=['x', 'y', 'z'])
            arc_lengths[pID] = float(im.iloc[-1, 0])

    return tls, starts, ends, n_missing, arc_lengths


def find_domain(x, y, z):
    """
    Given the positions, it returns a location identifier.
    """

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
    if type(x) in [float, int, np.float64]:
        for i, domain in enumerate(domains):
            xmin, xmax, ymin, ymax, zmin, zmax = domain
            if xmin < x and x < xmax and ymin < y and y < ymax and zmin < z and z < zmax:
                loc = i
    else:
        loc = np.ones_like(x) * -1  # initialize
        for i, domain in enumerate(tqdm(domains)):
            xmin, xmax, ymin, ymax, zmin, zmax = domain

            keep = np.greater_equal(x, xmin) * np.less(x, xmax) * \
                   np.greater_equal(y, ymin) * np.less(y, ymax) * \
                   np.greater_equal(z, zmin) * np.less(z, zmax)

            loc[keep] = i
    #             grps.append(1)
    #         else:
    #             grps.append(0)
    #     return grps
    return loc

    # def get_velocity(tr, start=0, end=None, deltaFrame=1, sample_period=100):
    frames = list(set(tr.frame))
    if end is None:
        end = np.nanmax(frames) - dFrame
    ux, uy, uz = [], [], []
    for k in tqdm(np.arange(start, end, sample_period), desc='extracting displacements'):
        rf = tp.relate_frames(tr, k, k + deltaFrame)
        # filtered trajectories
        rf = rf[~np.isnan(rf['x_b'])]
        #         x.append((rf['x'].values+rf['x_b'].values)/2.)
        #         y.append((rf['y'].values+rf['y_b'].values)/2.)
        # no filter on trajectories that don't exist
        # this keeps all existing points, which allows for matching with static crystal data
        # x.append(rf['x'].values)
        # y.append(rf['y'].values)
        #         ux.append(rf['dx'].values / deltaFrame)
        #         uy.append(rf['dy'].values / deltaFrame)
        #         uz.append(rf['dz'].values / deltaFrame)
        rf['ux'] = rf['dx'] / deltaT
        rf['uy'] = rf['dy'] / deltaT
        rf['uz'] = rf['dz'] / deltaT



def get_mapping_bw_pID_and_loc_no(tr, fns2track):
    """
    Returns a mapping function between particle ID and and the location identifier
    for particles that appear in the given frame(s) (fns2track)

    ... Returns a dictionary; dict[pID] = location_identifier
    """
    # Get all pIDs in the trajectory dataframe
    pIDs = np.asarray(list(set(tr.particle.to_numpy())))

    pIDs2track = []
    for fn in fns2track:
        subtr = tr[tr.frame == fn]
        pIDs_ = list(set(subtr.particle.to_numpy()))
        pIDs2track += pIDs_
        pIDs2track = list(set(pIDs2track))  # remove duplicates

    mapping_dict = {}
    for i, pID in enumerate(tqdm(pIDs2track, desc='Extracting loc_no to particles being tracked')):
        subtr = tr[tr.particle == pID]

        # First figure out where particles came from
        for fn in sorted(fns2track):
            pData = subtr[subtr.frame == fn]
            if len(pData) != 0:
                x, y, z = pData.x.to_numpy()[0], pData.y.to_numpy()[0], pData.z.to_numpy()[0]
                loc_no = find_domain(x, y, z)

                mapping_dict[pID] = loc_no
                break

    return mapping_dict


def gen_df_w_info_at_frame(tr, fn, mapping_pID_locNo=None, mappings_tr_stats=None, deltaFrame=1, fps=500.):
    """
    Generate a info
    """

    deltaT = deltaFrame / fps
    rf = tp.relate_frames(tr, fn, fn + deltaFrame, pos_columns=['x', 'y', 'z'])
    rf = rf[~np.isnan(rf['x_b'])]

    # Drop unnecessary columns
    rf = rf.drop(labels=['x_b', 'y_b', 'z_b'], axis=1)

    rf['ux'] = rf['dx'] / deltaT
    rf['uy'] = rf['dy'] / deltaT
    rf['uz'] = rf['dz'] / deltaT
    # Now drop ['dx'] etc.

    # rf uses rf.particle as their index- I don't want that
    rf = rf.reset_index()

    # Add velocity columns
    rf['ux'] = rf['dx'] / deltaT
    rf['uy'] = rf['dy'] / deltaT
    rf['uz'] = rf['dz'] / deltaT
    rf['umag'] = rf['dr'] / deltaT

    # Delete unnecessary columns
    rf = rf.drop(labels=['dx', 'dy', 'dz', 'dr'], axis=1)

    try:
        # Add location identifier (-1 if the particle does not exist in the specified frame(s))
        if not mapping_pID_locNo is None:
            rf['loc_no'] = rf.particle.map(mapping_pID_locNo)
            rf['loc_no'] = rf['loc_no'].replace(np.nan, -1).astype('int16')

        # Finally, if trail stats is available add those info to the dataframe
        if not mappings_tr_stats is None:
            tls, starts, ends, n_missing, arc_lengths = mappings_tr_stats
            rf['trail_length'] = rf.particle.map(tls)
            rf['start'] = rf.particle.map(starts)
            rf['end'] = rf.particle.map(ends)
            rf['n_missing'] = rf.particle.map(n_missing)
            rf['arc_length'] = rf.particle.map(arc_lengths)
        #     graph.pdf(rf.umag.to_numpy(), nbins=100)
    except:
        rf = None
    return rf


def save_df_in_h5(df, savepath, overwrite=True, verbose=True):
    """
    A wrapper to intelligently save a dataframe to a hdf5
    """

    pdir = os.path.split(savepath)[0]
    if not os.path.exists(pdir):
        os.makedirs(pdir)


    if not df is None:
        with h5py.File(savepath, mode='a') as f:
            cnames = df.columns
            for cname in cnames:
                if not cname in f.keys():
                    f.create_dataset(cname, data=df[cname].to_numpy())
                elif cname in f.keys() and overwrite:
                    del f[cname]
                    f.create_dataset(cname, data=df[cname].to_numpy())
                else:
                    if verbose:
                        print('... %s already exists in the file. Do not overwrite. Skipping...' % cname)
        if verbose:
            print('... a DataFrame is successfully formatted for Houdini and saved at ', savepath)
    else:
        # Create a dummy if DataFrame is empty
        with h5py.File(savepath, mode='a') as f:
            f.create_dataset('x', data=np.empty(0))
            f.create_dataset('y', data=np.empty(0))
            f.create_dataset('z', data=np.empty(0))
            f.create_dataset('particle', data=np.empty(0))
            pass
        print('... df is None. Empty hdf5 is created')



def get_pIDs_w_n_longest_tl(tr, n=1000, p=0):

    pIDs = list(set(tr.particle.to_numpy()))
    nf = len(list(set(tr.frame.to_numpy())))
    tl_thd = nf * p

    import time

    pIDs_cands = []
    tls = []
    for i, pID in enumerate(tqdm(pIDs, desc='get_pIDs_w_n_longest_tl')):
        t1 = time.time()
        subtr = tr[tr.particle == pID]
        t2 = time.time()
        # print('time took to extract a df for pID=%d: ' % pID, t2-t1)
        frames = subtr.frame.to_numpy()
        start = np.min(frames)
        end = np.max(frames)
        tl = end - start
        tls.append(tl)
        # if tl > tl_thd:
        #     pIDs_cands.append(pID)

    tls, pIDs_sorted = list(zip(*sorted(zip(tls, pIDs_cands))))

    return pIDs_sorted[:n]

def get_trail_stats_dict_faster(h5paths):
    h5paths = sorted(h5paths)

    starts = {}
    ends = {}
    tls = {}
    n_missing = {}
    n_detected = {}
    arc_lengths = {}

    x0s, y0s, z0s = {}, {}, {}

    pIDs_all = []
    for i, h5path in enumerate(tqdm(h5paths, desc='Extracting_trail_stats')):
        with h5py.File(h5path, mode='r') as f:
            pIDs = f['trackID'][...]
            xs, ys, zs = f['x'][...], f['y'][...], f['z'][...]

            for j, pID in enumerate(pIDs):
                if not pID in starts.keys():
                    starts[pID] = i
                    x0s[pID], y0s[pID], z0s[pID] = xs[j], ys[j], zs[j]
                    arc_lengths[pID] = 0.
                    n_detected[pID] = 1
                else:
                    dx, dy, dz = xs[j] - x0s[pID], ys[j] - y0s[pID], zs[j] - z0s[pID]
                    dr = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    arc_lengths[pID] += dr
                    n_detected[pID] += 1
                    # update x0, y0, z0 to compute arclength at the next step
                    x0s[pID], y0s[pID], z0s[pID] = xs[j], ys[j], zs[j]
                ends[pID] = i

            pIDs_all += list(pIDs)
            pIDs_all = list(set(pIDs_all))


    for i, pID in enumerate(tqdm(pIDs_all, desc='compute tls')):
        tls[pID] = ends[pID] - starts[pID] + 1
        n_missing[pID] = ends[pID] - starts[pID] + 1 - n_detected[pID]

    return tls, starts, ends, n_missing, arc_lengths

def get_mapping_bw_pID_and_loc_no_faster(h5paths, fns2track):
    mapping_dict = {}

    for i, fn in enumerate(tqdm(sorted(fns2track), desc='Extracting loc no. for pts in fns2track')):
        with h5py.File(h5paths[fn], mode='r') as f:
            x = f['x'][...]
            y = f['y'][...]
            z = f['z'][...]
            loc_no = find_domain(x, y, z)

            pIDs = f['trackID'][...]

        for j, pID in enumerate(pIDs):
            if not pID in mapping_dict.keys():
                mapping_dict[pID] = loc_no[j]
    return mapping_dict


def gen_df_w_info_at_frame_faster(h5dir, fn, mapping_pID_locNo=None, mappings_tr_stats=None,
                                  use_tp=True, deltaFrame=1, fps=500., nmin=0):
    """
    Generate a dataframe of points with related info about their trajectories at the frame number, fn in h5dir
    """
    if use_tp:
        df1 = generate_single_dframe(h5dir, fn, trackpy_format=True, mappings_tr_stats=mappings_tr_stats)
        df2 = generate_single_dframe(h5dir, fn+deltaFrame, trackpy_format=True, mappings_tr_stats=mappings_tr_stats)
        tr = pd.concat([df1, df2])


        deltaT = deltaFrame / fps

        rf = tp.relate_frames(tr, fn, fn + deltaFrame, pos_columns=['x', 'y', 'z'])
        rf = rf[~np.isnan(rf['x_b'])]
        # Drop unnecessary columns
        rf = rf.drop(labels=['x_b', 'y_b', 'z_b'], axis=1)

        rf['ux'] = rf['dx'] / deltaT
        rf['uy'] = rf['dy'] / deltaT
        rf['uz'] = rf['dz'] / deltaT
        # Now drop ['dx'] etc.

        # rf uses rf.particle as their index- I don't want that
        rf = rf.reset_index()

        # Add velocity columns
        rf['ux'] = rf['dx'] / deltaT
        rf['uy'] = rf['dy'] / deltaT
        rf['uz'] = rf['dz'] / deltaT
        rf['umag'] = rf['dr'] / deltaT

        # Delete unnecessary columns
        rf = rf.drop(labels=['dx', 'dy', 'dz', 'dr'], axis=1)

    else: # USE DAVIS DATA, and UPDATE IT WITH TRAIL STATS
        rf = generate_single_dframe(h5dir, fn, trackpy_format=True, nmin=nmin)

    try:
        # Add location identifier (-1 if the particle does not exist in the specified frame(s))
        if mapping_pID_locNo is not None:
            rf['loc_no'] = rf.particle.map(mapping_pID_locNo)
            rf['loc_no'] = rf['loc_no'].replace(np.nan, -1).astype('int16')

        # Finally, if trail stats is available add those info to the dataframe
        if mappings_tr_stats is not None:
            tls, starts, ends, n_missing, arc_lengths = mappings_tr_stats
            rf['trail_length'] = rf.particle.map(tls)
            rf['start'] = rf.particle.map(starts)
            rf['end'] = rf.particle.map(ends)
            rf['n_missing'] = rf.particle.map(n_missing)
            rf['arc_length'] = rf.particle.map(arc_lengths)
            # replace np.nan with -1
            rf['trail_length'] = rf['trail_length'].replace(np.nan, -1).astype('int16')
            rf['start'] = rf['start'].replace(np.nan, -1).astype('int16')
            rf['end'] = rf['end'].replace(np.nan, -1).astype('int16')
            rf['n_missing'] = rf['n_missing'].replace(np.nan, -1).astype('int16')
            rf['arc_length'] = rf['arc_length'].replace(np.nan, -1.0).astype('float')
        #     graph.pdf(rf.umag.to_numpy(), nbins=100)
    except:
        rf = None
    return rf




def main(h5dir,
         fns=None,
         search_radius=1.,  # umax = search_radius * fps (length unit in the h5 file/s)
         memory=0,
         fns2track=[90, 150, 220],
         npIDs2sample=10000,
         savedir=None,
         fps=500.,
         use_tp=False,
         deltaFrame=1,
         nmin=0,
         overwrite=False
         ):

    # Specify the location of data
    # h5dir = '/Users/takumi/Documents/research/turbulence/stb/tecplot_data/hdf5/l8p2mm_v200mms_f0p2hz_ftrig500hz_ext500us_10s_tr04_0575_1774_inc1'
    h5paths = sorted(glob.glob(os.path.join(h5dir, '*.h5')))

    # Create a dataframe (x,y,z, frame) from the raw data (h5)
    nfiles = len(h5paths)

    if fns is None:
        fns = np.arange(nfiles)
    start, end = min(fns), max(fns)

    if npIDs2sample is None:
        npIDs2sample_str = 'All'
    else:
        npIDs2sample_str = str(npIDs2sample)

    # The path to the directory where output will be stored
    dirname = os.path.split(h5dir)[1]
    param_str = 'frame%06d_%06d_ft' % (start, end) + '_'.join(map(str, fns2track)) + \
                '_sr%d_mem%d_nsamp%s' % (search_radius, memory, npIDs2sample_str)
    if savedir is None:
        savedir = os.path.join( os.path.join( os.path.split(h5dir)[0], 'trackpy'), dirname)
        savedir = os.path.join( savedir, 'processed_' + param_str)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # # Get location identifiers to track particles
    # fns2track = [90, 150, 220]  # user-defined parameter
    # # Sampling parameter
    # npIDs2sample = 10000 # number of pIDs to sample

    # Some useful lists
    top1 = [0, 200, 0, 200, -200, 0]  # top 1
    # top1 = [10, 200, 5, 200, -200, -10]  # top 1
    top2 = [-200, 0, 0, 200, -200, 0]  # top 2
    top3 = [-200, 0, 0, 200, 0, 200]  # top 3
    top4 = [0, 200, 0, 200, 0, 200]  # top 4
    bottom1 = [0, 200, -200, 0, -200, 0]  # bottom 1
    bottom2 = [-200, 0, -200, 0, -200, 0]  # bottom 2
    bottom3 = [-200, 0, -200, 0, 0, 200]  # bottom 3
    bottom4 = [0, 200, -200, 0, 0, 200]  # bottom 4


    # Processing
    if use_tp:
        savepath_tr = os.path.join(savedir, 'dataframe/dataframe_sradius%s_memory%d.h5' % (convert_float_to_decimalstr(search_radius), memory))
    else:
        savepath_tr = os.path.join(savedir, 'dataframe/dataframe_davis.h5')

    # Always load dataframe if it already exists
    if not os.path.exists(savepath_tr):
        if not os.path.exists(os.path.split(savepath_tr)[0]):
            os.makedirs(os.path.split(savepath_tr)[0])

        # Generate linked trajectories from particle positions using trackpy
        if use_tp:
            print('... use trackpy for linking detected particles')
            dfs = generate_dframes(h5dir, fns=fns)
            tr_all = pd.concat(tp.link_df_iter( dfs, search_radius, pos_columns=['x', 'y', 'z'], memory=memory) )
        # OR keep using DaVis outputs (x,y,z, ux,uy,uz, trackID)
        else:
            dfs = generate_dframes(h5dir, fns=fns, load_id=True, load_vel=True)
            tr_all = pd.concat(dfs)
        # save dataframe so that you dont have to repeat creating a dataframe
        tr_all.to_hdf(savepath_tr, key='df', mode='w')
    else:
        print('... Looking for trajectory dataframe (which stores all particle position throughout the movie' )
        print('...dataFrame found at ', savepath_tr)
        tr_all = pd.read_hdf(savepath_tr)


    # Trail stats
    savepath_tr = os.path.join(savedir, 'mapping_dicts/mappings_tr_stats.pkl')
    if os.path.exists(savepath_tr):
        print('... mapping functions for trail stats were found. Loading...')
        with open(savepath_tr, 'rb') as fp:
            mappings_tr_stats = pickle.load(fp)

        frames = list(set(tr_all.frame.to_numpy()))
        for frame in tqdm(frames):
            savepath = os.path.join(savedir, 'frame%05d.h5' % frame)
            if not os.path.exists(savepath):
                save_df_in_h5(tr_all[tr_all.frame == frame].rename(columns={'particle': 'trackID'}), savepath,
                              verbose=False)

    else:
        if not os.path.exists(os.path.split(savepath_tr)[0]):
            os.makedirs(os.path.split(savepath_tr)[0])
        if not use_tp:
            print('... Linking: stick to STB')
            print('... i.e. trackID assigned by STB will be used to get trail stats')
            mappings_tr_stats = get_trail_stats_dict_faster(h5paths)  # 50 min for 72k particles - this may not be super necessary
        else:
            print('... Linking: use trackpy')
            print('... i.e. trackID will be assigned by trackpy, and trail stats will be extracted based on the trajectories trackpy recognized.')
            # Extracting trail stats from a gigantic dataframe is TOO SLOW
            # ... it could take a day for 600k trajectories because it searches for individual particles EVERY TIME
            # Instead dump the dataframe in a formatted manner .
            print('... save linked trajectory data per frame (x, y, z, trackID)- these will be used and updated with trail stats')
            frames = list(set(tr_all.frame.to_numpy()))
            for frame in tqdm(frames):
                savepath = os.path.join(savedir, 'frame%05d.h5' % frame)
                save_df_in_h5(tr_all[tr_all.frame == frame].rename(columns={'particle': 'trackID'}) , savepath, verbose=False)
            h5paths = sorted(glob.glob(os.path.join(savedir, 'frame*.h5')))
            mappings_tr_stats = get_trail_stats_dict_faster(h5paths)

        print('... writing a pickle about trail stats')
        with open(savepath_tr, 'wb') as fp:
            pickle.dump(mappings_tr_stats, fp)

    # # Sample the trajectories if you think it will take too much time. Rule of thumb: 50 min for 10k trajectories
    # if not npIDs2sample is None:
    #     # Random sampling
    #     pIDs = list(set(tr_all.particle.to_numpy()))
    #     # pIDs_sample = random.sample(pIDs, npIDs2sample)
    #
    #     tls = mappings_tr_stats[0]
    #     pIDs_tmp = tls.keys()
    #     tl_values = tls.values()
    #     tl_values , pIDs_sample = list(zip(*sorted(zip(tl_values, pIDs_tmp), reverse=True)))
    #     pIDs_sample = pIDs_sample[:npIDs2sample]
    #
    #     # pIDs_sample = get_pIDs_w_n_longest_tl(tr_all, n=npIDs2sample) # not sorted
    #
    #     # tr = tr_all.loc[tr_all['particle'].isin(pIDs_sample)]
    #     print('... no of sampled trajectories / available trajectories: %d / %d (%.4f)' % (npIDs2sample, len(pIDs), npIDs2sample / len(pIDs)))
    # else:
    #     pIDs = list(set(tr_all.particle.to_numpy()))
    #     print('...  no of sampled trajectories / available trajectories: %d / %d (1.0)' % (len(pIDs), len(pIDs)))
    #     # tr = tr_all


    # Now extract useful information, and save it in h5 for every frame
    # location identifier
    savepath_loc = os.path.join(savedir, 'mapping_dicts/mapping_pID_locNo.pkl')
    if os.path.exists(savepath_loc):
        print('... mapping_pID_locNo was found. Loading...')
        with open(savepath_loc, 'rb') as fp:
            mapping_pID_locNo = pickle.load(fp)
    else:
        h5paths = sorted(glob.glob(os.path.join(savedir, 'frame*.h5')))
        mapping_pID_locNo = get_mapping_bw_pID_and_loc_no_faster(h5paths, fns2track) # 15 min for ~30k particles

        if not os.path.exists(os.path.split(savepath_loc)[0]):
            os.makedirs(os.path.split(savepath_loc)[0])
        with open(savepath_loc, 'wb') as fp:
            pickle.dump(mapping_pID_locNo, fp)

    print('No of tracked trajectories: %d / %d (%.5f %%)' % (len(mapping_pID_locNo.keys()),
                                                    len(mappings_tr_stats[0]),
                                                    len(mapping_pID_locNo.keys()) / len(mappings_tr_stats[0]) * 100) )


    # # Trail stats (This takes ridiculous amount of time)
    # savepath = os.path.join(savedir, 'mapping_dicts/mappings_tr_stats.pkl')
    # if os.path.exists(savepath):
    #     print('... mapping_pID_locNo found. \n'
    #           'Loading the existing file instead of creating it which could take many hours...')
    #     with open(savepath, 'rb') as fp:
    #         mappings_tr_stats = pickle.load(fp)
    # else:
    #     mappings_tr_stats = get_trail_stats_in_dict(tr)  # 50 min for 72k particles - this may not be super necessary
    #     with open(savepath, 'wb') as fp:
    #         pickle.dump(mappings_tr_stats, fp)

    # Finally, save data with relevant info every frame
    ## mapping 10k pts out of a list of 605k pts could take about 30 seconds! This is fast for humans, but it is not fast enough!
    ##
    def reduce_tr_mappings(mappings_tr_stats, nmin=10):
        print('... reducing the size of mapping functions, enabling fast turnout for houdini')
        tls, starts, ends, n_missing, arc_lengths = mappings_tr_stats

        tls_reduced = {pID: tl for pID, tl in tls.items() if tl >= nmin}
        starts_reduced, ends_reduced, n_missing_reduced, arc_lengths_reduced = {}, {}, {}, {}

        pIDs_reduced = tls_reduced.keys()

        starts_reduced = dict(zip(pIDs_reduced, list(map(starts.get, pIDs_reduced)  ) ) )
        ends_reduced = dict(zip(pIDs_reduced, list(map(ends.get, pIDs_reduced)   ) ) )
        n_missing_reduced = dict(zip(pIDs_reduced, list(map(n_missing.get, pIDs_reduced)   ) ) )
        arc_lengths_reduced = dict(zip(pIDs_reduced, list(map(arc_lengths.get, pIDs_reduced)   ) ) )

        print('... number of trajectories with greater than %d: %d /  %d (%.5f %%)'
              % (nmin, len(pIDs_reduced), len(tls), len(pIDs_reduced) / len(tls) * 100) )
        return tls_reduced, starts_reduced, ends_reduced, n_missing_reduced, arc_lengths_reduced

    print('... Mapping trail stats take a long time due to the number of short tracks (trail length ~ 1-4 frames)')
    print('... To avoid this, reduce the size of the mapping functions. Remove trail stats with trail length less than %d' % nmin )
    t0 = time.time()
    mappings_tr_stats = reduce_tr_mappings(mappings_tr_stats, nmin=nmin)
    t1 = time.time()
    print('time it took to reduce mapping functions by the number of trail length:', t1-t0)


    # Now modify h5 data in savedir
    h5paths = sorted(glob.glob(os.path.join(savedir, 'frame*.h5')))

    for i, fn in enumerate(tqdm(fns, desc='saving data every frame')):
        if fn + deltaFrame + 1 < len(h5paths):
            savepath = os.path.join(savedir, 'frame%05d.h5' % fn)
            if not os.path.exists(savepath) or overwrite:
                # rf = gen_df_w_info_at_frame(tr, fn, mapping_pID_locNo, mappings_tr_stats, fps=fps)
                rf = gen_df_w_info_at_frame_faster(savedir, fn, mapping_pID_locNo, mappings_tr_stats, fps=fps,
                                                   deltaFrame=deltaFrame, nmin=nmin)
                save_df_in_h5(rf, savepath, verbose=False)
    print('...savedir: ', savedir)
    print('... Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Adds a location index and compute ux, uy, uz to the existing data\n'
                    '... Location index: a number where particles originally camee from'
                    '...... assigned only for particles that appeared in the frames ones specify')

    # Locate Data
    parser.add_argument('-d', '--dir', type=str, default=None,
                        help='dir where h5 files are stored. h5 files should include AT LEAST x,y,z, trackID every frame\n'
                             'possibly x,y,z, ux,uy,uz, start, end, trail_length')
    parser.add_argument('-sd', '--savedir', type=str, default=None,
                        help='dir where h5 files are stored. h5 files should include AT LEAST x,y,z, trackID every frame\n'
                             'possibly x,y,z, ux,uy,uz, start, end, trail_length')
    parser.add_argument('-f', '--fns2track', nargs='+', help='frame numbers to specify particles to track',
                        default=[90, 150, 220], type=int)
    parser.add_argument('-s', '--start', type=int, help='use files in the args.dir from start to end',
                        default=0)
    parser.add_argument('-e', '--end', type=int, help='use files in the args.dir from start to end',
                        default=None)
    parser.add_argument('-n', '--nsample', type=int, help='Number of trajectories to sample throughout the movie',
                        default=None)
    parser.add_argument('-inc', '--inc', type=int, help='Frame increment, default: 1',
                        default=1)
    # Trackpy parameters
    parser.add_argument('-tp', '--use_trackpy', type=bool, help='If True, use trackpy to create trajectories. Otherwise, use DaVis trackIDs',
                        default=True)
    parser.add_argument('-sr', '--search_radius', type=float, help='maximum displacement- umax = search_radius * fps',
                        default=1.0)
    parser.add_argument('-m', '--memory', type=int, help='memory in trackpy',
                        default=0)
    parser.add_argument('-fps', '--fps', type=float, help='FPS- used to compute velocity',
                        default=500.)
    parser.add_argument('-nmin', '--nmin', type=int, help='minimum trajectory length to save',
                        default=1)

    parser.add_argument('-overwrite', '--overwrite', type=bool, help='overwrite the resulting h5',
                        default=True)


    # parser.add_argument('-comp_vel', '--comp_vel', type=bool, help='bool- If true, compute velocity for ALL particles',
    #                     default=False)
    args = parser.parse_args()

    print(args)
    nfiles = len(glob.glob(os.path.join(args.dir, '*.h5')))
    if args.end is None:
        args.end = nfiles
    fns = list(range(args.start, args.end, args.inc))

    if not all([element in fns for element in args.fns2track]):
        raise ValueError('fns2track includes integers which is not between [start, end] with increment of inc\n'
                         '... fns2track MUST be included in list(range(start, end, inc))\n'
                         '... The simplest solution would be setting inc of 1')

    main(h5dir=args.dir, fns=fns, fns2track=args.fns2track, savedir=args.savedir, npIDs2sample=args.nsample,
         search_radius=args.search_radius, memory=args.memory, fps=args.fps, use_tp=args.use_trackpy, nmin=args.nmin,
         overwrite=args.overwrite)