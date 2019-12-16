import glob
import numpy as np
from PIL import Image
import library.image_processing.movie as movies
import argparse
import os

'''
Make a flowtrace movie.
   - Sum n adjacent images every step.
Example usage:
python trace_flows.py -step 2 -ftm 25 -subtract_median -brighten 5 -overwrite -beta 0.8
'''


parser = argparse.ArgumentParser(description='Sum n images around each frame, and make a movie')
parser.add_argument('-check', '--check', help='Display intermediate results', action='store_true')
parser.add_argument('-overwrite', '--overwrite', help='Overwrite previous flow tracing results', action='store_true')
parser.add_argument('-brighten', '--brighten', help='Brighten all images by this factor', type=float, default=1.0)
parser.add_argument('-fps', '--fps', help='Frames per second of the generated movie', type=int, default=10)
parser.add_argument('-step', '--step', help='Sum the adjacement frames at every n step', type=int, default=10)
parser.add_argument('-ftm', '--ftm', help='Number of adjacement frames to sum/merge', type=int, default=30)
parser.add_argument('-subtract_median', '--subtract_median', help='Subtract median of images. Increasing brightness to ~5'
                                                                  'is recommended when you use this feature.', action='store_true')
parser.add_argument('-beta', '--beta', help='If you subtract median, you may choose to substract BETA*median. def=0.9', type=float, default=0.9)
parser.add_argument('-start', '--start', help=' def=0', type=int, default=0)
parser.add_argument('-end', '--end', help=' def=None', type=int, default=None)

args = parser.parse_args()
fps = args.fps

# Image dir
root = '/Volumes/labshared3-1/takumi/2018_07_27_fluorescence/'
indir = root + 'Tiff_folder/' \
               'PIV_fv_vp_left_macro55mm_fps2000_Dp56p57mm_D12p8mm_piston10p5mm_freq3Hz_v300mms_setting1_long/'
# Output dir
outdir = root + 'flowtrace_step%d_ftm%d_fps%d_subtractmed%r/' % (args.step, args.ftm, args.fps, args.subtract_median)

if not os.path.exists(outdir):
    os.mkdir(outdir)

tiffs = np.array(glob.glob(indir + '*.tiff'))
step = args.step
todo = np.arange(0, len(tiffs)-args.ftm, step)

# Get a median intensity
if args.subtract_median:
    print('Computing a median image')
    tiffs_considered = tiffs[todo]
    imsum_for_med = []
    for tiff in tiffs_considered:
        im = np.asarray(Image.open(tiff)).astype('float')
        imsum_for_med.append(im)
    im_med = (np.median(imsum_for_med, axis=0)*args.beta).astype('uint8')
    im_med[im_med > 255] = 255
    result = Image.fromarray(im_med)
    result.save(outdir + 'trace_flows_im_med' + '.png')
    print('... Done')



# If the frames are not already saved, or if we are to overwrite, go through and sum adjacent frames
if len(glob.glob(outdir + 'trace_flows*.png')) < len(todo) or args.overwrite:
    for (start, kk) in zip(todo, np.arange(len(todo))):
        print('start=' + str(start) + ', index = ' + str(kk) + '/' + str(len(todo)))
        end = start + args.ftm
        # initialize
        imsum = 0
        count = 0
        for tiff in tiffs[start:end]:
            im = np.asarray(Image.open(tiff)).astype('float')
            if args.subtract_median:
                im = im - im_med
            count += 1
            imsum += im
        imsum *= args.brighten / float(count)
        imsum = imsum.astype('uint8')
        imsum[imsum > 255] = 255


        result = Image.fromarray(imsum)
        result.save(outdir + 'trace_flows_{0:06d}'.format(kk) + '.png')

# Make movie
imgname = outdir + 'trace_flows_'
movname = root + 'flowtrace_step%d_ftm%d_fps%d_subtractmed%r' % (args.step, args.ftm, args.fps, args.subtract_median)
movies.make_movie(imgname, movname, indexsz='06', framerate=fps)
