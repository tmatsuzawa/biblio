import glob
import numpy as np
from PIL import Image
import library.image_processing.movie as movies
import argparse
import os
import cine
import PIL
import sys


'''
Make a flowtrace movie.
   - Sum n adjacent images every step.
Example usage:
python trace_flows.py -step 2 -ftm 25 -subtract_median -brighten 5 -overwrite -beta 0.8
'''

def fix_frame(raw_frame_data, bit_depth):
    """

    Parameters
    ----------
    raw_frame_data
    bit_depth: Cine.real_bpp

    Returns
    -------

    """
    #If the raw image has more than 8 bytes of data we need to fix this so we can
    #  save as a regular image!
    if cc.real_bpp > 8:
        raw_frame_data = (raw_frame_data >> (cc.real_bpp - 8)).astype('u1')
    return raw_frame_data

def change_contrast(imgarr, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)






parser = argparse.ArgumentParser(description='Sum n images around each frame, and make a movie')
parser.add_argument('-check', '--check', help='Display intermediate results', action='store_true')
parser.add_argument('-overwrite', '--overwrite', help='Overwrite previous flow tracing results', action='store_true')
parser.add_argument('-brighten', '--brighten', help='Brighten all images by this factor', type=float, default=1.0)
parser.add_argument('-fps', '--fps', help='Frames per second of the generated movie', type=int, default=10)
parser.add_argument('-step', '--step', help='Sum the adjacement frames at every n step', type=int, default=10)
parser.add_argument('-ftm', '--ftm', help='Number of adjacement frames to sum/merge', type=int, default=30)
parser.add_argument('-subtract_median', '--subtract_median', help='Subtract median of images. Increasing brightness to ~5'
                                                                  'is recommended when you use this feature.', action='store_true')
parser.add_argument('-subtract_mean', '--subtract_mean', help='Subtract mean of images. Increasing brightness to ~5'
                                                                  'is recommended when you use this feature.', action='store_true')
parser.add_argument('-subtract_first_im', '--subtract_first_im', help='Subtract the first image', action='store_true')
parser.add_argument('-beta', '--beta', help='Alpha for medioan/mean image. If you subtract median, you may choose to subtract BETA*median. def=0.9', type=float, default=0.9)

parser.add_argument('-start', '--start', help=' def=0', type=int, default=0)
parser.add_argument('-end', '--end', help=' def=None', type=int, default=None)

parser.add_argument('-cine', '--cine', help='', type=str,
                    default='/Volumes/bigraid/takumi/turbulence/3dprintedbox/PIV_Dp120p0mm_Do25p6mm/2019_01_16/' +
                            'PIV_fv_vp_left_macro55mm_fps200_Dp120p0mm_D25p6mm_piston7p9mm_freq5Hz_v400mms_setting1_inj1p0s_trig1p0s.cine')

args = parser.parse_args()

# Load cine
cc = cine.Cine(args.cine)  # cc is a Cine Class object

# Set basic variables
if args.end is None:
    end = len(cc)
else:
    end = args.end
if args.end > len(cc):
    print '... Invalid Entry: end you entered is greater than the total number of frames in a cine file.'
    sys.exit(1)
# Print some basic info
print '##########Info. about cine being processed######################################'
print 'Total frames:', len(cc)
print 'Bit depth:', cc.real_bpp
print 'Frame size: %d x %d' % (cc.width, cc.height)
print 'Frame rate:', cc.frame_rate
print '##########Info. about cine being processed######################################'

step = args.step
todo = np.arange(args.start, end-args.ftm, step)


# File architecture
root = os.path.splitext(args.cine)[0] + '/'
outdir = root + 'flowtrace_step%d_ftm%d_fps%d_subtractmed_%r_subtactmean%r/' % (args.step, args.ftm, args.fps, args.subtract_median, args.subtract_mean)
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Get a median/mean intensity
if args.subtract_median or args.subtract_mean:
    print '... Computing a median image'
    imarr = []
    im_med = np.empty((cc.width, cc.height))
    for frame in todo:
        raw = np.asarray(cc.get_frame(frame))
        raw = fix_frame(raw, cc.real_bpp)
        imarr.append(raw)
    if args.subtract_median:
        im_med = (np.median(imarr, axis=0) * args.beta).astype('uint8')
        im_med[im_med > 255] = 255
        result = Image.fromarray(im_med)
        result.save(outdir + 'trace_flows_im_med' + '.png')
    elif args.subtract_mean:
        im_mean = (np.mean(imarr, axis=0) * args.beta).astype('uint8')
        im_mean[im_mean > 255] = 255
        result = Image.fromarray(im_mean)
        result.save(outdir + 'trace_flows_im_mean' + '.png')
    print '... Done'
elif args.subtract_first_im:
    first_im = np.asarray(cc.get_frame(0))
    first_im = fix_frame(first_im, cc.real_bpp)


# If the frames are not already saved, or if we are to overwrite, go through and sum adjacent frames
if len(glob.glob(outdir + 'trace_flows*.png')) < len(todo) or args.overwrite:
    for (start, kk) in zip(todo, np.arange(len(todo))):
        print 'start=' + str(start) + ', index = ' + str(kk) + '/' + str(len(todo))
        end = start + args.ftm
        # initialize
        imsum = np.zeros((cc.height, cc.width))
        count = 0
        for frame in range(start, end):
            im = np.asarray(cc.get_frame(frame))
            im = fix_frame(cc.get_frame(frame), cc.real_bpp)
            if args.subtract_median:
                im = im - im_med
            elif args.subtract_mean:
                im = im - im_mean
            elif args.subtract_first_im:
                im = im - first_im
            count += 1
            imsum += im
        imsum *= args.brighten / float(count)
        imsum = imsum.astype('uint8')
        imsum[imsum > 255] = 255


        result = Image.fromarray(imsum)
        result.save(outdir + 'trace_flows_{0:06d}'.format(kk) + '.png')

# Make movie
imgname = outdir + 'trace_flows_'
movname = root + 'flowtrace_step%d_ftm%d_fps%d_subtractmed_%r_subtactmean%r' % (args.step, args.ftm, args.fps, args.subtract_median, args.subtract_mean)
movies.make_movie(imgname, movname, indexsz='06', framerate=args.fps)
