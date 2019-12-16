import glob
import numpy as np
from PIL import Image
import library.image_processing.movie as movies
import argparse
import os, sys
import cine
import PIL
import tqdm
import library.display.graph as graph
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage.filters as filters
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

def gamma(imgarr, gamma, bitdepth=8):
    int_max = 2 ** bitdepth - 1
    im = ((imgarr / int_max) ** (1 / gamma)) * int_max
    return im

def fix_contrast(imgarr, min=None, max=None, option='auto', ratio=0.8):
    if min is None:
        min = np.nanmin(imgarr)
    else:
        imgarr[imgarr < min] = min
    if max is None:
        max = np.nanmax(imgarr)
        imgarr[imgarr > max] = max

    if option=='enforce':
        bins, hist = compute_pdf(imgarr.flatten(), nbins=100)
        ind = np.argmax(np.cumsum(hist) * (bins[1]-bins[0]) > ratio)
        max = bins[ind]
        imgarr[imgarr > max] = max


    slope = 255./ (max - min)
    im = (imgarr - min) * slope
    return im

def apply_sobel_filter(im):
    """
    Applies Sobel operator (Edge detection filter)
    Standard ImageJ edge detection filter
    Parameters
    ----------
    im

    Returns
    -------
    im

    """
    derivative_x = np.array([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]]) / np.sqrt(2)
    derivative_y = np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]]) / np.sqrt(2)
    im1 = filters.convolve(im, derivative_x)
    im2 = filters.convolve(im, derivative_y)
    im = np.sqrt(im1 ** 2 + im2 ** 2)
    return im

def compute_pdf(data, nbins=10):
    # Get a normalized histogram
    # exclude nans from statistics
    hist, bins = np.histogram(data.flatten()[~np.isnan(data.flatten())], bins=nbins, density=True)
    # len(bins) = len(hist) + 1
    # Get middle points for plotting sake.
    bins1 = np.roll(bins, 1)
    bins = (bins1 + bins) / 2.
    bins = np.delete(bins, 0)
    return bins, hist

parser = argparse.ArgumentParser(description='Sum n images around each frame, and make a movie')

# Cine location
parser.add_argument('-cine', '--cine', help='', type=str,
                    default='/Volumes/bigraid/takumi/turbulence/3dprintedbox/PIV_Dp120p0mm_Do25p6mm/2019_01_16/' +
                            'PIV_fv_vp_left_macro55mm_fps200_Dp120p0mm_D25p6mm_piston7p9mm_freq5Hz_v400mms_setting1_inj1p0s_trig1p0s.cine')

# Flowtrace parameters
parser.add_argument('-brighten', '--brighten', help='Brighten all images by this factor', type=float, default=1.0)
parser.add_argument('-fps', '--fps', help='Frames per second of the generated movie', type=int, default=10)
parser.add_argument('-step', '--step', help='Sum the adjacement frames at every n step', type=int, default=10)
parser.add_argument('-ftm', '--ftm', help='Number of adjacement frames to sum/merge', type=int, default=30)
parser.add_argument('-gamma', '--gamma', help='Gamma contrast level', type=float, default=1)
parser.add_argument('-delta', '--delta', help='Contrast level parameter. Portion of data points included to maximize the contrast if enforce option is selected for fix_contrast()', type=float, default=0.9)


# Background correction
parser.add_argument('-subtract_median', '--subtract_median', help='Subtract median of images. Increasing brightness to ~5'
                                                                  'is recommended when you use this feature.', action='store_true')
parser.add_argument('-subtract_mean', '--subtract_mean', help='Subtract mean of images. Increasing brightness to ~5'
                                                                  'is recommended when you use this feature.', action='store_true')
parser.add_argument('-subtract_first_im', '--subtract_first_im', help='Subtract the first image', action='store_true')
parser.add_argument('-subtract_ref', '--subtract_ref', help='Subtract the ref', action='store_true')
parser.add_argument('-ref_cine', '--ref_cine', help='Reference image location',type=str,
                    default='/Volumes/bigraid/takumi/turbulence/3dprintedbox/PIV_Dp120p0mm_Do25p6mm/2019_01_17/Reference.cine')
parser.add_argument('-beta', '--beta', help='Alpha for medioan/mean image. If you subtract median, you may choose to subtract BETA*median. def=0.9', type=float, default=0.9)

# Options (Invert, Diff)
parser.add_argument('-invert', '--invert', help='Invert', action='store_true')
parser.add_argument('-edge', '--edge', help='Apply Sobel edge detection algorithm', action='store_true')
parser.add_argument('-diff', '--diff', help='Sum differences of successive images', action='store_true')
parser.add_argument('-diff_interval', '--diff_interval', help='If diff is True, take difference of two images at frame n and frame n + diff_interval. Default 5.',
                    type=int, default=5)
parser.add_argument('-uppercutoff', '--uppercutoff', help='Upper cutoff of raw image. im[im > uppercutoff] = uppercutoff',
                    type=int, default=255)




# Duration of movie
parser.add_argument('-start', '--start', help=' def=0', type=int, default=0)
parser.add_argument('-end', '--end', help=' def=None', type=int, default=None)

# Overwrite setting
parser.add_argument('-overwrite', '--overwrite', help='Overwrite previous flow tracing results', action='store_true')

args = parser.parse_args()

# Load cine
cc = cine.Cine(args.cine)  # cc is a Cine Class object

# Set basic variables
if args.end is None:
    end = len(cc)
else:
    end = args.end
if args.end > len(cc):
    print('... Invalid Entry: end you entered is greater than the total number of frames in a cine file.')
    sys.exit(1)
# Print some basic info
print('##########Info. about cine being processed######################################')
print('Total frames:', len(cc))
print('Bit depth:', cc.real_bpp)
print('Frame size: %d x %d' % (cc.width, cc.height))
print('Frame rate:', cc.frame_rate)
print('##########Info. about cine being processed######################################')

step = args.step
todo = np.arange(args.start, end-args.ftm, step)


# File architecture
root = os.path.splitext(args.cine)[0] + '/'
outdir = root + 'flowtrace_step%d_ftm%d_fps%d_diff%r_subtractmed_%r_subtactmean%r_subtactref%r/' % (args.step, args.ftm, args.fps, args.diff,  args.subtract_median, args.subtract_mean, args.subtract_ref)
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Get a median/mean intensity
if args.subtract_median or args.subtract_mean:
    print('... Computing a median image')
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
    print('... Done')
elif args.subtract_first_im:
    first_im = np.asarray(cc.get_frame(0))
    first_im = fix_frame(first_im, cc.real_bpp)
elif args.subtract_ref:
    cc_ref = cine.Cine(args.ref_cine)  # cc is a Cine Class object
    ref = np.asarray(cc_ref.get_frame(0))
    ref = fix_frame(ref, cc_ref.real_bpp)



# If the frames are not already saved, or if we are to overwrite, go through and sum adjacent frames
if len(glob.glob(outdir + 'trace_flows*.png')) < len(todo) or args.overwrite:
    for (start, kk) in zip(todo, np.arange(len(todo))):
        print('start=' + str(start) + ', index = ' + str(kk) + '/' + str(len(todo)))
        end = start + args.ftm
        # initialize
        imsum = np.zeros((cc.height, cc.width))
        count = 0
        for frame in range(start, end):
            im = np.asarray(cc.get_frame(frame))
            im = fix_frame(cc.get_frame(frame), cc.real_bpp)


            if args.diff:
                # graph.pdf(im, nbins=100)
                im_next = np.asarray(cc.get_frame(frame + args.diff_interval))
                im_next = fix_frame(cc.get_frame(frame + args.diff_interval), cc.real_bpp)
                im = im_next - im
                # graph.pdf(im_next, nbins=100)

            else:
                if args.subtract_median:
                    im = im - im_med
                elif args.subtract_mean:
                    im = im - im_mean
                elif args.subtract_first_im:
                    im = im - first_im
                elif args.subtract_ref:
                    im = im - ref

            count += 1
            imsum += im
            # graph.pdf(imsum, nbins=100, fignum=2)
        # Brighten
        im = imsum * args.brighten / float(count)
        # Contrast
        im = gamma(im, args.gamma)

        # Apply a Gaussian filter
        # im = gaussian_filter(im, sigma=2) # gaussian filter

        # Intensity cut
        im[im > args.uppercutoff] = args.uppercutoff
        # im[im < 50] = 0

        # Edge detection (Sobel filter)
        if args.edge:
            im = apply_sobel_filter(im)

        # Optimize contrast before saving
        im = fix_contrast(im, min=np.nanmedian(im)/2., option='enforce', ratio=args.delta) # this might be useful but this will not perform uniform contrast adjustment for all frames
        # im = fix_contrast(im, min=np.nanmedian(im) / 2.,  max=255) # this is recommended since it will perform uniform contrast adjustment for all frames

        im = im.astype('uint8')
        result = Image.fromarray(im)
        result.save(outdir + 'trace_flows_{0:06d}'.format(kk) + '.png')
# Close cine file
cc.close()

# Make movie
imgname = outdir + 'trace_flows_'
movname = root + 'flowtrace_step%d_ftm%d_fps%d_diff%r_subtractmed_%r_subtactmean%r_subtactref%r_invert%r'\
                 % (args.step, args.ftm, args.fps, args.diff, args.subtract_median, args.subtract_mean, args.subtract_ref, args.invert)
movies.make_movie(imgname, movname, indexsz='06', framerate=args.fps, overwrite=args.overwrite, invert=args.invert)
