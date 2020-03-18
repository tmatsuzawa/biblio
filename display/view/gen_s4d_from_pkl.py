#!/urs/bin/env python
from numpy import *
import os, argparse, sys, re
import cine
import pickle
import glob
import numpy as np
from tqdm import tqdm

def natural_sort(arr):
    def atoi(text):
        'natural sorting'
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        '''
        natural sorting
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [atoi(c) for c in re.split('(\d+)', text)]

    return sorted(arr, key=natural_keys)


def get_intensity_from_RBGA(data_tmp):
    """
    Returns intensity from a RGBA vector
    (R, G, B, alpha) = (0.0-255.0, 0.0-255.0, 0.0-255.0, 0.0-1.0)

    Parameters
    ----------
    data_tmp

    Returns
    -------

    """
    intensity = np.zeros(data_tmp.shape[:-1])
    for i in range(3):
        intensity += data_tmp[..., i] ** 2
    intensity = np.sqrt(intensity / 3)
    return intensity

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert numpy 3D arrays to S4D')
    parser.add_argument('input', metavar='input', type=str, nargs='+', default=None,
                        help='Directory containing pickles to be converted into an S4D')
    parser.add_argument('-z', dest='z', type=int, default=119, help='volume dimension')
    parser.add_argument('-y', dest='y', type=int, default=130, help='volume dimension')
    parser.add_argument('-x', dest='x', type=int, default=200, help='volume dimension')
    parser.add_argument('-nf', dest='nf', type=int, default=1, help='Number of previous frames to superimpose. Must be > 0.')
    parser.add_argument('-tau', dest='tau', type=int, default=3, help='Half-life of alpha; Alpha decays exponentially as the frames get older.')
    parser.add_argument('-start', dest='start', type=int, default=0, help='The first frame number loaded to s4d- Default: 0')
    parser.add_argument('-end', dest='end', type=int, default=None, help='The last frame number loaded to s4d- Default: None')
    parser.add_argument('-inc', dest='inc', type=int, default=1, help='Increment to load pickle files to s4d- Default: 1')
    parser.add_argument('-RGBA', dest='RGBA', default=True, help='The last frame number loaded to s4d- Default: None')
    parser.add_argument('-mode', dest='mode', default='maxint', help='Option to assign intensity in the voxel. "sum" or "maxint" ')
    args = parser.parse_args()

    pickle_directories = args.input
    z = args.z
    y = args.y
    x = args.x
    nf = args.nf
    start, end, inc = args.start, args.end, args.inc
    RGBA = args.RGBA

    if RGBA:
        ncomp = 4
    else:
        ncomp = 1
    shape = (nf, z, y, x, ncomp)


    # Iterate through inputs
    for pd in pickle_directories:
        pickle_fyles = sorted(glob.glob(os.path.join(pd, '*.pickle')))
        pickle_fyles = natural_sort(pickle_fyles)
        # Total frames of the s4d object
        if end is None:
            end = len(pickle_fyles)

        # Initialize S4D container
        header = {
            'bottom_clip': 0,
            'dtype': 'u1',
            'frame size': (z, y),
            'original_bitdepth': 12,
            'use_3dsetup_perspective': False,
            '3dsetup': '#3d setup file\n'
                       '#Required fields: \'cine_depth\', \'display_frames\', \'bottom_clip\', \'u1_top\', \'u1_gamma\', \'scale\', \'rate\', \'x_func\', \'y_func\', \'z_func\'\n\n'
                       '#-------------------------------------------------------------------------------\n'
                       '#For converting from cine/sparse to S4D\n'
                       'cine_depth = %d\n'
                       'x_size = %d\n'
                       'frame_shape = (%d, %d)\n'
                       'display_frames = range(1, %d + 1)\n'
                       'bottom_clip = 80\n\n'
                       '#For conversion to 8 bit only\nu1_top = 4000\nu1_gamma = 2.0\n\n'
                       '#-------------------------------------------------------------------------------\n\n'
                       '# Scale is mm/pixel\n'
                       'scale = 0.585\n'
                       '# Rate is volumes/sec\n'
                       'rate = 16E6/210/452\n'
                       '# Number of volumes passed before start\n'
                       'volume_delay = 300\n\n'
                       '#===============================================================================\n'
                       '#measure and input these values\n\n'
                       '#width of tank\n'
                       'W = 16.575\n\n'
                       '#z_sign\n'
                       'z_sign = 1\n\n'
                       '#front of the tank positions\n'
                       'front_top =  27.45/2.54\n'
                       'front_bottom = 46.9/2.54\n\n'
                       '#back of the tank positions\n'
                       'back_top = 33.6/2.54\n'
                       'back_bottom = 59.3/2.54\n\n'
                       '#camera info\n'
                       'f = 55.\n'
                       'pix_size = 28 * 10**-3\n\n'
                       '#calculate relevant input parameters\n\n'
                       'a = x_size * scale / 2.\na_in = a / 25.4\n\n'
                       '#--------------------------------------\n'
                       'h1 = back_top - front_top\n'
                       'h2 = back_bottom - front_bottom\n'
                       'theta_1 = atan(h1 / W)\n'
                       'theta_2 = atan(h2 / W)\n\n\n'
                       'L = ((pix_size + scale) / (pix_size) * f) / a * 1.33\n\n'
                       '# print "a      : ", a\n# print "a_in   : ", a_in\n'
                       '# print "Theta 1: ", theta_1*180./pi\n'
                       '# print "Theta 2: ", theta_2*180./pi\n'
                       '# print "L      : ", L\n\n'
                       'mt = h1 / W\n'
                       'mb = h2 / W\n\n'
                       'x_x  = 1\n'
                       'x_y  = 0\n'
                       'x_z  = 0\n'
                       'x_xy = 0\n'
                       'x_xz = -1 / L\n\n'
                       '# print "x :", x_x, x_y, x_z, x_xy, x_xz\n\n'
                       'y_y  = 1\n'
                       'y_x  = 0\n'
                       'y_z  = 0 \n'
                       'y_xy = 0\n'
                       'y_yz = -1 / L\n\n'
                       '# print "y :", y_x, y_y, y_z, y_xy, y_yz\n\n'
                       'z_z  = 1\n'
                       'z_y  = -1./2 * (mt + mb)\n'
                       'z_x  = 0\n'
                       'z_yz = 1./2 * (mb - mt)\n'
                       'z_xz = 0\n\n'
                       '# print "z :", z_x, z_y, z_z, z_xz, z_yz\n\n'
                       '# x/y/z scaled (-1:1) for cubic frame\n'
                       '# def\'s are used instead of lambda to get closure on the variables.\n\n'
                       'def x_func(x, y, z, x_x=x_x, x_y=x_y, x_z=x_z, x_xy=x_xy, x_xz=x_xz):\n'
                       '    return x_x*x + x_y*y + x_z*z + x_xy*x*y + x_xz*x*z\n\n'
                       'def y_func(x, y, z, y_y=y_y, y_x=y_x, y_z=y_z, y_xy=y_xy, y_yz=y_yz):\n'
                       '    return y_x*x + y_y*y + y_z*z + y_xy*x*y + y_yz*y*z \n\n'
                       'def z_func(x, y, z, z_z=z_z, z_x=z_x, z_y=z_y, z_xz=z_xz, z_yz=z_yz):\n'
                       '    return z_z*z + z_y*y + z_x*x + z_yz*y*z + z_xz*x*z' %
                       (x, x, z, y, x)
        }

        output_fn = pd + '_start%03d_end%03d_inc%03d_nf%03d_tau%03d_%s.s4d' % (start, end, args.inc, args.nf, args.tau, args.mode)
        output = cine.Sparse4D(output_fn, 'w', header)
        tau0 = args.tau / np.log(2) # intensity becomes half after 3 frames

        # for i, pf in enumerate(pickle_fyles[start:end]):
        #     print("loading data for frame ", (i+start))
        #     for j in range(i-args.nf, i+1):
        #         if j >= 0:
        #             data_tmp = pickle.load(open(pickle_fyles[j], 'rb'))data_arr
        #             if j == 0:
        #                 data = np.zeros_like(data_tmp)
        #
        #             # Add blurring effect
        #             data_tmp[0][-1] *= np.exp(- (i - j) / tau)
        #
        #             data += data_tmp
        #
        #     output.append_array(data)
        #
        #     print("completed frame: ", (i+start))
        #
        # output.close()


        for i, pf in enumerate(tqdm(pickle_fyles[start:end-nf:inc])):
            if i == 0:
                data_arr = np.empty(shape)
                intensity = np.empty((nf, z, y, x))
                for j in range(0, nf):
                    if not RGBA:
                        data_arr[j, ..., 0] = pickle.load(open(pickle_fyles[i * inc + j], 'rb'))
                    else:
                        # Reinitialize data_arr
                        data_arr = np.empty(shape)
                        data_arr[j, ...] = pickle.load(open(pickle_fyles[i * inc + j], 'rb'))
                        intensity[j, ...] = get_intensity_from_RBGA(data_arr[j, ...])
            else:
                if RGBA:
                    # Reset alpha for all frames in data_arr
                    for j in range(0, nf):
                        data_arr[j, ..., -1] = 1
                # Roll the array- now the oldest frame is in data_arr[-1, ...]
                data_arr = np.roll(data_arr, -1, axis=0)
                # Replace the oldest frame with the newest frame
                data_arr[-1, ...] = pickle.load(open(pickle_fyles[i*inc+nf-1], 'rb'))

            if RGBA:
                # Change alpha for old frames otherwise do nothing
                for j in range(0, nf):
                    data_arr[j, ..., -1] *= np.exp(- (nf - 1 - j) / tau0)

            if args.mode == 'sum':
                # Sum over frames
                data = np.nansum(data_arr, axis=0)
            elif args.mode == 'maxint':
                # Assign the maximum intensity for each voxel instead of adding them together
                ind = np.argmax(intensity, axis=0)
                a0 = np.repeat(ind[..., np.newaxis], ncomp, axis=-1)
                a1, a2, a3, a4 = np.indices(a0.shape)
                data = data_arr[a0, a1, a2, a3, a4]
            else:
                print('... args.mode was not valid! By default, assign the maximum intensity over adjacent frames for each voxel.')
                data = np.nansum(data_arr, axis=0)


            output.append_array(data)
        output.close()

        print('... Done! s4d object is saved at')
        print(output_fn)
