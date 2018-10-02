import argparse
import cine
from scipy import misc
import os
import library.basics.formatstring as fs
import numpy as np
import tqdm


def cine2tiff(cinefile, step, start=0, end=None, ctime=1, folder='/Tiff_folder', post='', imroot='im'):
    """
    Generate a list of tiff files extracted from a cinefile.
        Different modes of processing can be used, that are typically useful for PIV processings :
        test : log samples the i;ages, using the function test_sample. Default is 10 intervals log spaced, every 500 images.
        Sample : standard extraction from start to stop, every step images, with an interval ctime between images A and B.
        File : read directly the start, stop and ctime from a external file. Read automatically if the .txt file is in format :
        'PIV_timestep'+cine_basename+.'txt'
    INPUT
    -----
    file : str
        filename of the cine file
    mode : str.
        Can be either 'test','Sample', 'File'
        single : list of images specified
        pair : pair of images, separated by a ctime interval
    step : int
        interval between two successive images to processed.
    start : int. default 0
        starting index
    end : int. default 0.
        The cine will be processed 'till its end
    ctime :
    folder : str. Default '/Tiff_folder'
        Name of the root folder where the images will be saved.
    post : str. Default ''
        post string to add to the title of the tiff folder name
    OUTPUT
    OUTPUT
    None
    """
    # file : path of the cine file

    try:
        c = cine.Cine(cinefile)
    except:
        print('Cine file temporary unavailable')
        return None

    print('Length : ' + str(len(c)))
    cinefile_short = fs.get_filename_wo_ext(cinefile)
    savedir = os.path.dirname(cinefile) + folder + '/' + cinefile_short + '/'
    if not os.path.exists(savedir):
        try:
            os.mkdir(savedir)
        except OSError:
            os.mkdir(os.path.dirname(cinefile) + folder)
            os.mkdir(savedir)
    if end is None:
        end = len(c)
    frames = np.arange(start, end, step)
    for i in tqdm.trange(len(frames), desc='cine2tiff_step%d'%step):
        frame = frames[i]
        filename = savedir + imroot + "%06d" %frame + '.tiff'
        if not os.path.exists(filename):
            imarr = c.get_frame(frame)  # 2d array at each frame
            misc.imsave(filename, imarr, 'tiff')
    print '... Done'



parser = argparse.ArgumentParser(description="Make tiffs from a cine")
parser.add_argument('-f',dest='filepath',default=None,type=str,help='full path to a cinefile')
parser.add_argument('-step',dest='step',default=10,type=int, help='Sampling step of the cine. Default value is 10')
parser.add_argument('-start',dest='start',default=0,type=int, help='Frame number from which tiff files are created')
parser.add_argument('-end',dest='end',default=None,type=int, help='Frame number to which tiff files are created')

args = parser.parse_args()



if __name__ == '__main__':
    cine2tiff(args.filepath, args.step, start=args.start)