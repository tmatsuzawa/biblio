#!/usr/bin/env python
import argparse
import json
import os
import pickle
import sys
import time
import glob
import sparse
from fmm import *
from ilpm import path, vector
from matplotlib import cm
from numpy import *
import matplotlib.pyplot as plt


cmap = cm.viridis
LINE_STYLES = ['solid', 'dashed', 'dotted']

HYDROFOIL_CODEX_JSON = None
if not HYDROFOIL_CODEX_JSON:
    DATA_ROOT = os.path.expanduser('~/hydrofoils')
    if not os.path.exists:
        os.mkdir(DATA_ROOT)
    HYDROFOIL_CODEX_JSON = os.path.join(DATA_ROOT, 'HYDROFOIL_CODEX.json')

# SEEDS_PER_PATH = 5
LINE_TRACE_STEP = 0.5
PC = True
SCALED = False
EPSILON = 0.05


class Clicker:
    """A Class used to store information about click positions and display them on a figure."""

    def __init__(self, ax):
        self.canvas = ax.get_figure().canvas
        self.cid = None
        self.pt_lst = []
        self.pt_plot = ax.scatter([], [], marker='o', color='m')
        self.connect_sf()

    def clear(self):
        """Clears the points"""
        self.pt_lst = []
        self.redraw()

    def connect_sf(self):
        if self.cid is None:
            self.cid = self.canvas.mpl_connect('button_press_event', self.click_event)

    # def disconnect_sf(self):
    #     if self.cid is not None:
    #         self.canvas.mpl_disconnect(self.cid)
    #         self.cid = None

    def click_event(self, event):
        """ Extracts locations from the user"""
        if event.key == 'shift':
            # Shift + click to remove all selections
            print('cleared click events')
            self.clear()
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            print('click event: xdata = %f, ydata= %f' % (event.xdata, event.ydata))
            self.pt_lst.append((event.xdata, event.ydata))
        elif event.button == 3:
            # Right click to remove selected pt
            print('removed click event near: data = %f, ydata = %f' % (event.xdata, event.ydata))
            self.remove_pt((event.xdata, event.ydata))

        self.redraw()

    def remove_pt(self, loc):
        """ Removes point from pt_lst that is nearest to loc"""
        if len(self.pt_lst) > 0:
            self.pt_lst.pop(argmin([sqrt((x[0] - loc[0]) ** 2 + (x[1] - loc[1]) ** 2) for x in self.pt_lst]))

    def redraw(self):
        """ Scatter points from pt_lst onto the figure"""
        if len(self.pt_lst) > 0:
            pts = asarray(self.pt_lst)
        else:
            pts = []

        self.pt_plot.set_offsets(pts)
        self.canvas.draw()

def Power(x, n): return x ** n
def Sqrt(x): return x ** (1. / 2.)
def atan(x): return arctan(x)


def check_encoder_info(sparse_file):
    """
    Checks for existence of processed encoder speed files. If they don't exist they are generated.
    Parameters
    ----------
    sparse_file : str
        The sparse file_name.

    """
    base, fyle = os.path.split(sparse_file)
    name, ext = os.path.splitext(fyle)
    name_parts = name.split('_')
    yyyy_mm_dd = name_parts[0] + '_' + name_parts[1] + '_' + name_parts[2]
    cart_info_dir = os.path.join(base, yyyy_mm_dd)

    if not os.path.isfile(os.path.join(cart_info_dir, name + '_position_data_info.pickle')):
        os.system('python process_encoder_speeds.py ' + cart_info_dir)


def check_for_path(sparse_file, frame):
    """Check to see if the path exists for the given frame.

    Parameters
    ----------
    sparse_file : str
        The sparse file name.
    frame : int
        The desired frame.

    Returns
    -------
    exists : bool
        True if it exists, false otherwise
    """

    fn = get_path_name(frame, sparse_file)
    if os.path.exists(fn):
        return True
    else:
        return False


def check_trace(lines_xyz):
    """Check for an improperly generated tangle object

    Parameters
    ----------
    lines_xyz : array
        Nx3 array of points from which a Tangle will be generated

    Returns
    -------
    exists : bool
        True if lines_xyz is sensible, false otherwise
    """

    t = path.Tangle(lines_xyz)
    for p in t:
        if math.isnan(p.L):
            return False
    return True


def check_trace_with_statistics(sparse_file):
    """Update the log after the first round of connecting points, using the length variance to select
        which frames were not traced well. Also confirm that tangles were generated properly by check that path
        lengths are not NaN of Infinity

    Parameters
    ----------
    sparse_file : str
        The sparse file_name.
    """

    log_path = os.path.splitext(sparse_file)[0] + '_log.json'
    with open(log_path, 'r') as log_file:
        data = json.load(log_file)
    # frames = []
    # lengths = []
    for f in data['length']:
        # frames.append(f)
        # lengths.append(data['length'][f])

        if len(data['frames']):
            last_good_frame = None
            i = 0
            while last_good_frame is None:
                if data['frames'][str(int(f) - i)] == 'good':
                    last_good_frame = str(int(f) - i)
                i += 1
        else:
            data['frames'][f] = 'good'

        path_diff = (data['length'][f] - data['length'][last_good_frame]) / data['length'][last_good_frame]
        if abs(path_diff) > 0.1:
            data['frames'][f] = 'bad'
        else:
            data['frames'][f] = 'good'

    out_file = open(os.path.splitext(sparse_file)[0] + '_log.json', 'w')
    json.dump(data, out_file)
    out_file.close()
    # sigma = std(lengths)
    # n = 6
    # windowed_mean = []
    # for i, l in enumerate(lengths):
    #     if i < n:
    #         windowed_mean.append(mean(lengths[0:n-1]))
    #     elif i > len(lengths) - n:
    #         windowed_mean.append(mean(lengths[-n:-1]))
    #     else:
    #         windowed_mean.append(mean(lengths[i-(n/2):i+(n/2)]))
    #
    # for i, fl in enumerate(zip(frames, lengths)):
    #     if windowed_mean[i] - 2*sigma < fl[1] < windowed_mean[i] + 2*sigma:
    #         update_log(sparse_file, fl[0], 'frames', 'good')
    #     else:
    #         update_log(sparse_file, fl[0], 'frames', 'bad')
    #
    # path_dir = get_sparse_subdir(sparse_file, 'paths')
    # tangles = glob.glob(os.path.join(path_dir, '*.json'))
    # for t in tangles:
    #     frame = str(int(t.split('_')[-1][0:3]))
    #     T = path.load_tangle(t)
    #     for p in T:
    #         if math.isnan(p.L) or math.isinf(p.L):
    #             update_log(sparse_file, frame, 'frames', 'bad')


def directed_fast_march(raw_data, seed_points, SEEDS_PER_PATH):
    lines = []
    N = len(seed_points)

    if N / SEEDS_PER_PATH != int(N / SEEDS_PER_PATH):
        print("WARNING -- NUMBER OF SEED POINTS PER PATH MUST BE %d" % SEEDS_PER_PATH)

    Npaths = N / SEEDS_PER_PATH

    for j in range(Npaths):

        loop = []

        for i in range(SEEDS_PER_PATH):

            source = seed_points[i + j * SEEDS_PER_PATH]
            target = seed_points[(i + 1) % SEEDS_PER_PATH + j * SEEDS_PER_PATH]

            zr, yr, xr = [x.flatten() for x in mgrid[-1:2, -1:2, -1:2]]
            rr = array([xr, yr, zr]).T
            tp = vstack([t + rr for t in [target]])

            dist_map = msfm3d(raw_data + 10E-5, source, terminate=4E5, term_points=tp, usesecond=False, usecross=False,
                              tp_error=False)

            line = trace_line(dist_map, target, source, step=LINE_TRACE_STEP, max_steps=4000)

            if line is not None:
                line = line[::-1]
            else:
                return []

            loop.append(line[:-1])

        lines.append(vstack(loop))

    return lines


def filter_raw_data(S4D, frame, blur=1.0):
    """Blur the S4D volume to make sure that fast marching doesn't get stuck.

    Parameters
    ----------
    S4D : an S4D object
        The S4D object containing the volume you want to blur.
    frame : int
        The frame of the volume you want to blur.
    blur : float, optional (default = 1.0)
        The sigma value for the gaussian blur.

    Returns
    -------
    raw_data : [X,Y,Z] array (floats)
        An array of the blurred intensity values of the volume from the S4D.
    """

    raw_data = S4D[frame].astype('f')
    raw_data *= 1. / raw_data.max()
    raw_data = ndimage.filters.gaussian_filter(raw_data, blur)

    return raw_data


def fmt_time(t):
    """Converts time to a human readable format"""
    return '%dhr:%02dmin:%02dsec' % (int(t / 3600.), int(t / 60.) % 60, int(t) % 60)


def get_sparse_subdir(sparse_name, dirname):
    """Get a subdir inside the main sparse data directory. Will make the requested dir if
        it doesn't exist.

    Parameters
    ----------
    sparse_name : str
        The file name of the sparse for the experiment
    dir_name : str
        The name of the subdir.

    Returns
    -------
    subdir : str
        The full path for the subdir.
    """

    name = os.path.splitext(sparse_name)[0]
    if not os.path.exists(name): os.mkdir(name)
    subdir = os.path.join(name, dirname)
    if not os.path.exists(subdir): os.mkdir(subdir)
    return subdir


def get_loop_length(loop_xyz):
    """

    Parameters
    ----------
    loop_xyz

    Returns
    -------

    """
    return sum(vector.mag(vector.plus(loop_xyz) - loop_xyz))


def get_path_name(frame, sparse_file):
    """Get the path file name for a given frame and sparse file.

    Parameters
    ----------
    frame : int
        The frame
    sparse_file : str
        The sparse file name.

    Returns
    -------
    path_fn : str
        The file name for the path .json file
    """

    path_dir = get_sparse_subdir(sparse_file, 'paths')
    base, fyle = os.path.split(sparse_file)
    name = os.path.splitext(fyle)[0]

    return os.path.join(path_dir, name + '_paths_%03d.json' % frame)


def get_total_loop_length(loops_xyz):
    """

    Parameters
    ----------
    loops_xyz

    Returns
    -------

    """
    lengths = list(map(get_loop_length, loops_xyz))
    return sum(lengths)


def make_log(sparse_file):
    """Make a log file for the given experiment to record the progress and success/failure
        of the tracing algorithm.

    Parameters
    ----------
    sparse_file : str
        The sparse file_name.
    """

    if not os.path.isfile(os.path.splitext(sparse_file)[0] + '_log.json'):
        data = {'info': os.path.splitext(sparse_file)[0], 'frames': {}, 'length': {}}
        out_file = open(os.path.splitext(sparse_file)[0] + '_log.json', 'w')
        json.dump(data, out_file)
        out_file.close()

        print('log generated for ' + sparse_file[:-7])


def perspective_correct(lines_xyz, setup_info):
    """Perspective correct multiple paths at once.

    Parameters
    ----------
    lines_xyz : list ([N,3] arrays of floats)
        A list of all line segments represented by the ordered (x,y,z) points on the paths.
    setup_info : str
        The 3dsetup information from the s4d header.

    Returns
    -------
    lines_xyz : list ([N,3] array of floats)
        A list of perspetive corrected paths.
    """

    for i, l in enumerate(lines_xyz): lines_xyz[i] = perspective_invert(l, setup_info)
    return lines_xyz


def perspective_invert(X, setup_info, suppress_output=True):
    """Correct for the perspective of the camera and laser scanner.

    Parameters
    ----------
    X : [N,3] array
        A closed path that you want to persepctive correct.
    setup_info : str
        The 3dsetup information from the s4d header.

    Returns
    -------
    corrected_X : [N,3] array
        The perspective corrected path.
    """

    if suppress_output:
        setup_info = setup_info.replace('print ', '')

    exec(setup_info)

    re_X = X - (frame_shape[0] / 2., frame_shape[1] / 2., x_size / 2.)
    re_X = re_X / (x_size / 2.)
    xp = re_X[..., 0]
    yp = re_X[..., 1]
    zp = re_X[..., 2]

    x = (xp * (-(x_xz * y_yz * zp) + x_xz * yp * z_yz + x_xz * y_y * z_z - 2 * x_x * y_yz * z_z +
               x_xz * Sqrt(4 * y_yz * (y_y * zp - yp * z_y) * z_z + Power(-(y_yz * zp) + yp * z_yz + y_y * z_z, 2)))
         ) / (2. * (Power(x_xz, 2) * (y_y * zp - yp * z_y) - Power(x_x, 2) * y_yz * z_z +
                    x_x * x_xz * (-(y_yz * zp) + yp * z_yz + y_y * z_z)))

    y = (y_yz * zp - yp * z_yz + y_y * z_z - Sqrt(4 * y_yz * (y_y * zp - yp * z_y) * z_z +
                                                  Power(-(y_yz * zp) + yp * z_yz + y_y * z_z, 2))) / (
            2 * y_yz * z_y - 2 * y_y * z_yz)

    z = (y_yz * zp - yp * z_yz - y_y * z_z + Sqrt(4 * y_yz * (y_y * zp - yp * z_y) * z_z +
                                                  Power(-(y_yz * zp) + yp * z_yz + y_y * z_z, 2))) / (2. * y_yz * z_z)

    pc_X = asarray([x, y, z]).T

    return pc_X * (x_size / 2.) + (frame_shape[0] / 2., frame_shape[1] / 2., x_size / 2.)


def plot_lengths(sparse_file):
    path_dir = get_sparse_subdir(sparse_file, 'paths')
    tangles = glob.glob(os.path.join(path_dir, '*.json'))
    tangles.sort()
    lengths = []
    frames = []
    ticks = []
    for i, tangle in enumerate(tangles):
        t = path.load_tangle(tangle)
        lengths.append(t.L)
        frame = int(tangle.split('_')[-1].split('.json')[0])
        frames.append(frame)
        plt.scatter(frame, t.L)
        if i != 0:
            if abs(t.L[0] - lengths[i - 1][0]) / t.L[0] > 0.1:
                ticks.append(frame)
    plt.plot(frames, lengths)
    plt.xticks(ticks)
    plt.savefig(os.path.join(path_dir, 'lengths.png'))
    plt.close()


def propagate_seed_points(raw_data, old_seed_points, window=3):
    """
    Determine location of seed points in the next frame of an S4D based upon brightness and previous seed locations
    Parameters
    ----------
    raw_data : [X,Y,Z] array (floats)
        An array of the blurred intensity values of the volume from the S4D
    old_seed_points : [X,Y,Z] array (ints)
        Locations of seed points in an S4D
    window : int
        Size of the window around each element in old_seed_points. This is passed to argmax_neighborhood which then
        selects the brightest pixel in the window to be the propagated seed point.

    Returns
    -------
    [X,Y,Z] array (ints)
        Locations of seed points in an S4D
    """
    xs, ys = old_seed_points[:, 0], old_seed_points[:, 1]
    width = (raw_data.shape[0] / 2, window, window)
    z, y, x = asarray([argmax_neighborhood(raw_data, (width[0], x[1], x[0]), width) for x in zip(xs, ys)],
                      dtype='float').T

    return asarray([x, y, z]).T


def select_seed_points(raw_data, axis=0, fig_size=10, window=3):
    """
    GUI for selecting seed points
    Parameters
    ----------
    raw_data : [X,Y,Z] array (floats)
        An array of the blurred intensity values of the volume from the S4D
    axis : int
        The axis you want to project the 3D raw_data onto
    fig_size : int
        Dimension of the square figure the raw_data is projected onto
    window : int
        Size of the window around each mouse-click event. The brightest pixel in each window
        is then taken to be a seed point.

    Returns
    -------
    [X,Y,Z] array (ints)
        Locations of seed points in an S4D
    """
    fig = plt.figure(figsize=(fig_size, fig_size))
    fig.add_subplot(111)
    plt.imshow(raw_data.max(axis), cmap=cmap)
    plt.axis('off')
    guidepost = Clicker(plt.gca())
    plt.show()

    xs, ys = list(zip(*guidepost.pt_lst))
    width = (raw_data.shape[0] / 2, window, window)
    z, y, x = asarray([argmax_neighborhood(raw_data, (width[0], x[1], x[0]), width) for x in zip(xs, ys)],
                      dtype='float').T

    return asarray([x, y, z]).T


def save_snapshot(s4d, frame, lines_xyz, sparse_file):
    """Saves a projection of the completed trace.

    Parameters
    ----------
    s4d : s4d object
        The s4d object containing all the volumes from the experiment.
    frame : int
        The frame of the s4d.
    lines_xyz : list ([N,3] arrays of floats)
        Closed loops.
    sparse_file : str
        The file name of the sparse file for the experiment.
    """

    proj_dir = get_sparse_subdir(sparse_file, 'path_proj')
    sparse_name = os.path.splitext(os.path.split(sparse_file)[-1])[0]
    fn = os.path.join(proj_dir, sparse_name + '_paths_%03d.png' % frame)

    plt.figure(figsize=(10, 10))
    plt.imshow(s4d[frame].sum(0), cmap=cmap)
    plt.axis('off')
    for i, X in enumerate(lines_xyz):
        plt.plot(X[:, 0], X[:, 1], color='r', linestyle=LINE_STYLES[i], lw=1)

    plt.gcf()
    plt.savefig(fn, bbox_inches='tight')
    plt.close()


def save_trace(frame, lines_xyz, setup_info, sparse_file,
               info=['PC', 'scaled', 'scale', 'rate', 'cart_speed', 'cart_acceleration', 'cart_acc_time',
                     'cart_stroke_len']):
    """Save the trace as a tangle object in a .json file with optional additional info stored in the info attribute.
        **This contains an algorithm for reorienting paths so the moment points in the direction of travel**
    Parameters
    ----------
    frame : int
        The frame for the volume
    lines_xyz : list ([N,3] arrays of floats)
        A list of all line segments represented by the ordered (x,y,z) points on the paths.
    setup_info : str
        The 3dsetup information from the s4d header.
    sparse_file : str
        The file name of the sparse file.
    info : list (strings), optional (default = ['PC', 'scaled', 'scale', 'rate', 'cart_speed', 'cart_acceleration', 'cart_acc_time',
                     'cart_stroke_len'])
        Fields for the optional information. In addition to the default information, 'hydrofoil_info' is also recognized if information
        about the hydrofoil used to generate the path should be stored in the .json. Note that this requires that there be a corresponding
        entry in the hydrofoil codex.
    """

    T = path.Tangle(lines_xyz)
    for p in T:
        if p.moment()[2] < 0:
            p.path = p.path[::-1]

    if 'PC' in info:
        T.info['PC'] = PC
    if 'scaled' in info:
        T.info['scaled'] = SCALED

    exec (setup_info)

    if 'scale' in info:
        T.info['scale'] = scale
    if 'rate' in info:
        T.info['rate'] = rate

    base, fyle = os.path.split(sparse_file)
    name, ext = os.path.splitext(fyle)
    name_parts = name.split('_')
    yyyy_mm_dd = name_parts[0] + '_' + name_parts[1] + '_' + name_parts[2]
    cart_info_dir = os.path.join(base, yyyy_mm_dd)
    pickle_file = os.path.join(cart_info_dir, name + '_position_data_info.pickle')
    info_dict = pickle.load(open(pickle_file))

    if 'cart_speed' in info: T.info['cart_speed'] = info_dict['speed']
    if 'cart_acceleration' in info: T.info['cart_acceleration'] = info_dict['acceleration']
    if 'cart_acc_time' in info: T.info['cart_acc_time'] = info_dict['acc_time']
    if 'cart_stroke_len' in info: T.info['cart_stroke_len'] = info_dict['stoke_len']

    if 'hydrofoil_info' in info:
        hydrofoil_codex = json.load(open(HYDROFOIL_CODEX_JSON))
        hydrofoil_code = name_parts[3]
        if hydrofoil_code in list(hydrofoil_codex.keys()):
            hydrofoil_info = hydrofoil_codex[hydrofoil_code]
            for k, v in zip(list(hydrofoil_info.keys()), list(hydrofoil_info.values())):
                T.info['hydrofoil_' + k] = v
            T.info['hydrofoil_code'] = hydrofoil_code

    fn = get_path_name(frame, sparse_file)
    T.save(fn)


def update_log(sparse_file, frame, key, status):
    """Update the tracing log

    Parameters
    ----------
    sparse_file : str
        The sparse file_name.
    frame : int
        The frame.
    key : str
        The key for the feature.
    status : str
        The status, perhaps the length or ('good' or 'bad'), depending on the key
    """
    log_path = os.path.splitext(sparse_file)[0] + '_log.json'
    with open(log_path, 'r') as log_file:
        data = json.load(log_file)
    data[key][frame] = status
    os.remove(log_path)
    with open(log_path, 'w') as log_file:
        json.dump(data, log_file)


def organize_lines(lines_xyz, sparse_file, f, kind):
    """Sort loops found from fast marching with loops from the previous frames

    Parameters
    ----------
    lines_xyz : list ([N,3] arrays of floats)
        Closed loops.
    sparse_file : str
        The sparse file name.
    frame : int
        The frame
    kind : str
        The sorting method

    Returns
    -------
    lines_xyz : list ([N,3] arrays of floats)
        Closed loops.
    """
    i = 1
    while i < 11:
        fn = get_path_name(f - i, sparse_file)
        if os.path.exists(fn):
            print('i == ', i)
            T_last = path.load_tangle(fn)
            if len(T_last) != len(lines_xyz):
                i += 1
                continue
            lines = list(lines_xyz)
            lines = perspective_correct(lines, setup_info)
            T_current = path.Tangle(lines)
            if kind == 'Writhe':
                print('Checking Writhe Diffs')
                writhe_diffs = [T_current.crossing_matrix()[0].sum(1)[0] - wr for wr in
                                T_last.crossing_matrix()[0].sum(1)]
                if writhe_diffs[0] < writhe_diffs[1]:
                    return lines_xyz
                else:
                    return lines_xyz[::-1]
            if kind == 'Length':
                print('Checking Length Diffs')
                len_diffs = [T_current.L[0] - l for l in T_last.L]
                if len_diffs[0] < len_diffs[1]:
                    return lines_xyz
                else:
                    return lines_xyz[::-1]
            if kind == 'Center':
                print('Checking Center Diffs')
                center_diffs = [T_current[0].center() - c for c in [p.center() for p in T_last]]
                r1 = sqrt(asarray([p ** 2 for p in center_diffs[0]]).sum())  # print 'zero ', center_diffs[0].sum()
                r2 = sqrt(asarray([p ** 2 for p in center_diffs[1]]).sum())  # print 'one ', center_diffs[1].sum()
                if r1 < r2:
                    return lines_xyz
                else:
                    return lines_xyz[::-1]
            else:
                print('Sorting Method Not Specified!')
                return lines_xyz
        else:
            i += 1
            continue

    print('No Comparison Path Within Range')
    return lines_xyz


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automated but not robust line tracing. Uses ridge points as input')
    parser.add_argument('input', metavar='input', type=str, nargs='+', default=None, help='.sparse files')
    parser.add_argument('-frames', dest='frames', type=str, nargs='+', default=None,
                        help='Frames to be processed. Use the standard python slice notation. One input with multiple'
                             'sparse files will apply that frame range to every sparse. Multiple inputs will apply the'
                             'frame range to the respective sparse file.')
    parser.add_argument('-seeds', dest='seeds', type=int, default=5, help='number of seed points per path')
    parser.add_argument('-sort_key', dest='sort_key', type=str, default='Writhe', help='key for sorting paths')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true', default=False,
                        help='Overwrite previously traced paths')

    # START TIMER
    start = time.time()

    # PARSE COMMAND LINE INPUTS
    args = parser.parse_args()
    sparse_files = args.input
    seeds = args.seeds
    kind = str(args.sort_key)
    overwrite = args.overwrite

    # PARSE WHETHER OR NOT THE FRAMES ARE THE SAME FOR EACH SPARSE FILE
    frames = []
    if len(args.frames) == 1 and ',' not in args.frames[0]:
        for sparse_file in sparse_files:
            frames.append(eval('x[%s]' % args.frames[0], {'x': list(range(1000))}))
    elif len(args.frames) == len(sparse_files) and ',' not in args.frames[0]:
        for f in args.frames:
            frames.append(eval('x[%s]' % f, {'x': list(range(1000))}))
    elif ',' in args.frames[0]:
        for sparse_file in sparse_files:
            temp_frames = []
            for f in args.frames[0].split(','):
                temp_frames.append(int(f))
            frames.append(temp_frames)
    else:
        sys.exit('Number of input sparse files does not match number of frames options')

    # PROCESS ENCODER INFO IF IT DOESN'T ALREADY EXIST
    check_encoder_info(sparse_files[0])

    # LOAD .S4D FILES AND SELECT INITIAL SEED POINTS VIA GUI
    initial_seed_points = []

    for sparse_file in sparse_files:
        s4d_file = sparse_file[:-7] + '.s4d'
        s4d = sparse.Sparse4D(s4d_file)
        raw_data = filter_raw_data(s4d, frames[0][0])
        seed_points = select_seed_points(raw_data)
        initial_seed_points.append(seed_points)
    for i, sparse_file in enumerate(sparse_files):

        # LOAD THE SETUP INFO
        s4d_file = os.path.splitext(sparse_file)[0] + '.s4d'
        s4d = sparse.Sparse4D(s4d_file)
        setup_info = s4d.header['3dsetup']

        # GENERATE THE LOG FILE
        # make_log(sparse_file)

        # PROPAGATE SEED POINTS AND TRACE PATHS
        propagated_seed_points = [initial_seed_points[i]]
        for j, f in enumerate(frames[0]):

            # CHECK TO SEE IF THE PATH EXISTS FOR THIS FRAME
            path_existence = check_for_path(sparse_file, f)
            print('Working on Frame ', f)
            if j > 0:
                k = frames[0][j-1] + 1
                while f > k:
                    print('Propagating points through frame ', k)
                    raw_data = filter_raw_data(s4d, k)
                    new_seed_points = propagate_seed_points(raw_data, propagated_seed_points[-1])
                    propagated_seed_points.append(new_seed_points)
                    k += 1

            if not path_existence or (path_existence and overwrite):

                # ATTEMPT TO TRACE PATH
                raw_data = filter_raw_data(s4d, f)
                new_seed_points = propagate_seed_points(raw_data, propagated_seed_points[-1])
                propagated_seed_points.append(new_seed_points)
                lines = directed_fast_march(raw_data, propagated_seed_points[-1], seeds)

                # LOG BAD FRAME IF A CLOSED PATH COULDN'T BE TRACED
                if not lines:
                    # update_log(sparse_file, f, 'frames', 'bad')
                    print('No Closed Path Found')

                # SAVE IF A CLOSED PATH WAS FOUND
                else:
                    if len(lines) > 1:
                        lines = organize_lines(lines, sparse_file, f, kind)
                    save_snapshot(s4d, f, lines, sparse_file)
                    lines = perspective_correct(lines, setup_info)
                    if check_trace(lines):
                        save_trace(f, lines, setup_info, sparse_file)
                    else:
                        attempt = 1
                        while attempt < 3:
                            print('Attempt: ' + str(attempt))
                            lines = directed_fast_march(raw_data, propagated_seed_points[-1] + attempt * EPSILON, seeds)
                            if len(lines) > 1:
                                lines = organize_lines(lines, sparse_file, f, kind)
                            save_snapshot(s4d, f, lines, sparse_file)
                            lines = perspective_correct(lines, setup_info)
                            if check_trace(lines):
                                save_trace(f, lines, setup_info, sparse_file)
                                break
                            else:
                                attempt += 1
                    # path_length = get_total_loop_length(lines)
                    # update_log(sparse_file, f, 'length', path_length)
                    print('Closed Path Traced')

            # SKIP FRAME IF PATH EXISTS AND OVERWRITE WAS NOT SPECIFIED
            else:
                # raw_data = filter_raw_data(s4d, f)
                # new_seed_points = propagate_seed_points(raw_data, propagated_seed_points[-1])
                # propagated_seed_points.append(new_seed_points)
                print('Path Already Traced... Skipping Frame ', f)
                continue

        # UPDATE LOG
        # check_trace_with_statistics(sparse_file)
        plot_lengths(sparse_file)
        print('Finished Tracing in Sparse File ', sparse_file)
    print('Done in %s' % fmt_time(time.time() - start))
