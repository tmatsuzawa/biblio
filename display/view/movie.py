#!/usr/bin/env python
import numpy as np
import v4d_shader as v4d
from scipy.interpolate import interp1d
import sys, os
from ilpm.vector import mag, norm
import argparse
from cine import Sparse4D
import subprocess

# Abs path to ffmpeg
ffmpeg_path = '/Users/takumi/Documents/git/takumi/library/library/image_processing/ffmpeg'


def gram_schmidt(V):
    V = list(np.array(v, dtype='d') for v in V)
    U = []

    for v in V:
        v2 = v.copy()

        for u in U:
            v -= np.dot(v2, u) * u

        U.append(norm(v))

    return np.array(U)


def normalize_basis(V):
    V = gram_schmidt(V[:2])

    # Ensures right handed coordinate system
    V = np.vstack((V, np.cross(V[0], V[1])))

    return V


def interp_basis(b1, b2, sub_divides=4, zero_tol=0.01):
    B = [gram_schmidt(b1[:2]), gram_schmidt(b2[:2])]

    if np.sqrt(((B[1] - B[0]) ** 2).sum()) < 1E-6:
        return lambda x: normalize_basis(B[1])

    for i in range(sub_divides):
        B2 = []

        for j, b in enumerate(B):
            B2.append(b)

            if (j + 1) < len(B):
                b2 = B[j + 1]

                bm = (b + b2) / 2.
                m = mag(bm)

                z = np.cross(b[0], b[1])
                if (m < zero_tol).all():
                    bm[0] += 2 * z * zero_tol
                    bm[1] += 2 * b[0] * zero_tol
                elif m[0] < zero_tol:
                    bm[0] += 2 * z * zero_tol
                elif m[1] < zero_tol:
                    bm[1] += 2 * z * zero_tol

                B2.append(gram_schmidt(bm))

        B = B2

    B = list(map(normalize_basis, B))

    d = [0]

    for i in range(len(B) - 1):
        dB = B[i] - B[i + 1]
        d.append(d[-1] + np.sqrt((dB ** 2).sum()))

    interp = interp1d(np.array(d), np.array(B), axis=0)
    return lambda x: normalize_basis(interp(x * d[-1]))


def strip_comments(s):
    if '#' in s:
        s = s[:s.find('#')]
    return s


def unpack_commands(s):
    s = s.replace('\t', ' ' * 8)
    lines = list(filter(str.strip, map(strip_comments, s.splitlines())))

    commands = []
    while lines:
        line = lines.pop(0).strip()
        if ':' in line:
            cmd, arg = map(str.strip, line.split(':', 1))
        else:
            cmd, arg = line, ''

        kwargs = {}
        while lines and lines[0].startswith(' '):
            line = lines.pop(0).strip()
            if ':' in line:
                key, val = map(str.strip, line.split(':', 1))
            else:
                key, val = line, ''
            kwargs[key.lower()] = val

        commands.append((cmd.lower(), arg, kwargs))

    return commands


# Use eval_kwargs instead!
# def check_kwargs(d, special=[]):
#    for k in d.keys():
#        if k not in v4d.valid_movie_options and k not in special:
#            raise ValueError('invalid movie frame keyword "%s"' % k)


def eval_kwargs(d):
    for k, v in d.items():
        if k in v4d.valid_movie_options:
            d[k] = v4d.valid_movie_options[k](eval(v))
        else:
            raise ValueError('invalid movie frame keyword "%s"' % k)


rot_d = {
    'x': v4d.rot_x,
    'y': v4d.rot_y,
    'z': v4d.rot_z
}


def find_previous(s, k):
    for step in reversed(s):
        if k in step:
            return step[k]
    else:
        raise ValueError('movie keyword "%s" must be previously defined to execute a sequence.  '
                         '(e.g. it should appear in a "single" command before "steps")' % k)


def make_movie(imgname=None, imgdir=None, movname=None, indexsz='05', framerate=10, rm_images=False,
               save_into_subdir=False, start_number=0, framestep=1, ext='png', option='normal', overwrite=False,
               invert=False, add_commands=[]):
    """Create a movie from a sequence of images using the ffmpeg supplied with ilpm.
    Options allow for deleting folder automatically after making movie.
    Will run './ffmpeg', '-framerate', str(int(framerate)), '-i', imgname + '%' + indexsz + 'd.png', movname + '.mov',
         '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])

    ... ffmpeg is not smart enough to recognize a pattern like 0, 50, 100, 150... etc.
        It tries up to an interval of 4. So 0, 3, 6, 9 would work, but this hinders practicality.
        Use the glob feature in that case. i.e. option='glob'

    Parameters
    ----------
    imgname : str
        ... path and filename for the images to turn into a movie
        ... could be a name of directory where images are stored if option is 'glob'
    movname : str
        path and filename for output movie (movie name)
    indexsz : str
        string specifier for the number of indices at the end of each image (ie 'file_000.png' would merit '03')
    framerate : int (float may be allowed)
        The frame rate at which to write the movie
    rm_images : bool
        Remove the images from disk after writing to movie
    save_into_subdir : bool
        The images are saved into a folder which can be deleted after writing to a movie, if rm_images is True and
        imgdir is not None
    option: str
        If "glob", it globs all images with the extention in the directory.
        Therefore, the images does not have to be numbered.
    add_commands: list
        A list to add extra commands for ffmpeg. The list will be added before output name
        i.e. ffmpeg -i images command add_commands movie_name
        exmaple: add_commands=['-vf', ' pad=ceil(iw/2)*2:ceil(ih/2)*2']
    """
    # if movie name is not given, name it as same as the name of the img directory
    if movname is None:
        if os.path.isdir(imgname):
            if imgname[-1] == '/':
                movname = imgname[:-1]
            else:
                movname = imgname
        else:
            pdir, filename = os.path.split(imgname)
            movname = pdir


    if not option=='glob':
        command = [ffmpeg_path,
                   '-framerate', str(int(framerate)),
                   '-start_number', str(start_number),
                   '-i', imgname + '%' + indexsz + 'd.' + ext,
                   '-pix_fmt', 'yuv420p',
                   '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12', '-threads', '0', '-r', '100']
    else:
        # If images are not numbered or not labeled in a sequence, you can use the glob feature.
        # On command line,
        # ffmpeg -r 1
        # -pattern_type glob
        # -i '/Users/stephane/Documents/git/takumi/library/image_processing/images2/*.png'  ## It is CRITICAL to include '' on the command line!!!!!
        # -vcodec libx264 -crf 25  -pix_fmt yuv420p /Users/stephane/Documents/git/takumi/library/image_processing/images2/sample.mp4
        command = [ffmpeg_path,
                 '-pattern_type', 'glob',  # Use glob feature
                 '-framerate', str(int(framerate)),  # framerate
                 '-i', imgname + '/*.' + ext,  # images
                 '-vcodec', 'libx264',  # codec
                 '-crf', '12',  # quality
                 '-pix_fmt', 'yuv420p']
    if overwrite:
        command.append('-y')
    if invert:
        command.append('-vf')
        command.append('negate')
    # check if image has dimensions divisibly by 2 (if not ffmpeg raises an error... why ffmpeg...)
    # ffmpeg raises an error if image has dimension indivisible by 2. Always make sure that this is not the case.
    # image_paths = glob.glob(imgname + '/*.' + ext)
    # img = mpimg.imread(image_paths[0])
    # height, width = img.shape
    # if not (height % 2 == 0 and width % 2 == 0):
    command += ['-vf', ' pad=ceil(iw/2)*2:ceil(ih/2)*2']


    print(command)
    command += add_commands

    command.append(movname + '.mp4')
    subprocess.call(command)

    # Delete the original images
    if rm_images:
        print('Deleting the original images...')
        if not save_into_subdir and imgdir is None:
            imdir = os.path.split(imgname)
        print('Deleting folder ' + imgdir)
        subprocess.call(['rm', '-r', imgdir])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View a 4D image.')
    parser.add_argument('input', metavar='input', type=str, help='text file created from v4d_shader or s4d')
    parser.add_argument('-f', dest='frames', type=str, default="50",
                        help='range of frames to display, in python slice format or single frame')
    parser.add_argument('-b', dest='brightness', type=float, default=2., help='brightness of images')
    parser.add_argument('-z', dest='zoom', type=float, default=4.0, help='zoom setting on box')
    parser.add_argument('-v', dest='angle', type=str, default='45',
                        help='sets viewing angle (either side, 45, face, or up')
    parser.add_argument('-o', dest='out', type=str, default=None, help='directory for images')
    parser.add_argument('--stereo', dest='stereo', action='store_true', default=False,
                        help='add stereoscopic anaglyph feature')
    parser.add_argument('--no_box', dest='no_box', action='store_true', default=False, help='remove bounding box')
    parser.add_argument('--no_persp', dest='no_persp', action='store_true', default=False,
                        help='remove 3d perspective from images')
    parser.add_argument('--frate', dest='frate', default=10,
                        help='Frame rate of the movie')
    # parser.add_argument('--3d', dest='3d', action='store_true', default=False,
    #                     help='add stereoscopic anaglyph feature')
    # parser.add_argument('--3d', dest='3d', action='store_true', default=False,
    #                     help='add stereoscopic anaglyph feature')

    args = parser.parse_args()

    # sequence = [{'r':v4d.rot_y(a, eye(3)[:2]), 'frame':30, 'z':2, '3d':True, 'brightness':.5} for a in 2 * pi * arange(30)/ 30]

    # v4d.make_movie('2011_10_19_square_240fpv_250vps_B.s4d', sequence, 'test_movie')
    input = args.input
    OUTDIR, file_type = os.path.splitext(input)
    movname = OUTDIR + '_movie.mp4'

    # if not os.path.exists(movname):
        # if file_type == '.txt':
        #
        #     source = None
        #     sequence = []
        #     window_kwargs = {}
        #
        #     for cmd, arg, kwargs in unpack_commands(open(input).read()):
        #         if cmd == 'source':
        #             source = arg
        #
        #         if cmd == 'size':
        #             w, h = map(int, arg.split(','))
        #             window_kwargs['width'] = w
        #             window_kwargs['height'] = h
        #
        #         elif cmd == 'single':
        #             eval_kwargs(kwargs)
        #             sequence.append(kwargs)
        #
        #         elif cmd == 'steps':
        #             if 'spin' in kwargs:
        #                 axes, rotations = kwargs['spin'].split(',')
        #                 axes = axes.lower()
        #                 rotations = float(rotations)
        #
        #                 if axes in rot_d:
        #                     spin_func = lambda x, R: rot_d[axes](x * 2 * np.pi * rotations, R)
        #                 else:
        #                     raise ValueError('spin option should have value: (xyz), number of spins')
        #
        #                 del kwargs['spin']
        #             else:
        #                 spin_func = None
        #
        #             eval_kwargs(kwargs)
        #
        #             if spin_func is not None and 'r' not in kwargs:
        #                 kwargs['r'] = find_previous(sequence, 'r')
        #
        #             N = int(eval(arg))
        #             steps = [{} for n in range(N)]
        #
        #             for k, v in kwargs.items():
        #                 # print k, type(v)
        #                 if type(v) in (list, tuple):
        #                     v = np.array(v)
        #
        #                 if k == 'r':
        #                     r0 = find_previous(sequence, 'r')
        #                     interp_func = interp_basis(r0, v)
        #                     for i in range(N):
        #                         x = (i + 1) / float(N)
        #                         R = interp_func(x)
        #
        #                         if spin_func is not None:
        #                             R = spin_func(x, R)
        #
        #                         steps[i]['r'] = R
        #
        #                 elif type(v) in (float, np.ndarray) or k == 'frame':
        #                     v0 = find_previous(sequence, k)
        #                     for i in range(N):
        #                         x = (i + 1) / float(N)
        #                         steps[i][k] = x * v + (1 - x) * v0
        #
        #                 else:
        #                     steps[0][k] = v
        #
        #             sequence += steps
        #     # for s in sequence:
        #     print(sequence)
        #
        # elif file_type == '.s4d':
        #     source = input
        #     window_kwargs = {}
        #
        #     if ':' in args.frames:
        #         frames = range(*[int(x) for x in args.frames.split(':')])
        #         angle = args.angle
        #         brightness = args.brightness
        #         zoom = args.zoom
        #         no_persp = args.no_persp
        #         no_box = args.no_box
        #         stereo = args.stereo
        #         S4D = Sparse4D(input)
        #
        #         if no_persp:
        #             fov = 0
        #             z = 2.82842712475
        #         else:
        #             fov = 45
        #             z = 4.0
        #
        #         if angle == 'side':  # add fov for 3d perspective
        #             sequence = [{'r': v4d.rot_y(-np.pi / 2., np.eye(3)), 'frame': float(f),
        #                          'z': z, '3d': stereo, 'brightness': brightness, 'box': not no_box, 'fov': no_persp}
        #                         for f in frames]
        #             OUTDIR += '_SIDE'
        #         elif angle == 'front':
        #             sequence = [{'r': v4d.rot_y(np.pi, np.eye(3)), 'frame': float(f),
        #                          'z': z, '3d': stereo, 'brightness': brightness, 'box': not no_box, 'fov': no_persp}
        #                         for f in frames]
        #             OUTDIR += '_FRONT'
        #         elif angle == 'top':
        #             sequence = [{'r': v4d.rot_x(np.pi / 2, np.eye(3)), 'frame': float(f),
        #                          'z': z, '3d': stereo, 'brightness': brightness, 'box': not no_box, 'fov': no_persp}
        #                         for f in frames]
        #             OUTDIR += '_TOP'
        #         elif angle == '45':
        #             sequence = [{'r': v4d.rot_y(-130. * np.pi / 180., np.eye(3)), 'frame': float(f),
        #                          'z': 4.0, '3d': stereo, 'brightness': brightness, 'box': not no_box} for f in frames]
        #             OUTDIR += '_ANGLE'
        #         else:
        #             raise ValueError("Specified viewing angle may be either '45', 'side', 'front', or 'top'!")
        #     else:
        #         frame = int(args.frames)
        #         sequence = [{'r': v4d.rot_y(a, np.eye(3)[:2]), 'frame': frame, 'z': 4.0, '3d': False, 'brightness': 2.}
        #                     for a in 2 * np.pi * np.arange(30) / 30]
        #         OUTDIR += '_SPIN_FRAME%s' % int(frame)
        # else:
        #     raise ValueError('Input type should be s4d file or text file generated from vd4_shader!')
        #
        # # CORRECTION FOR PYTHON3 (for some reason 1st frame ends up being all black) this is not the best way around it
        # # but it gets the job done...
        # sequence.insert(0, sequence[0])
        # if args.out is not None:
        #     OUTDIR = args.out
        #
        # v4d.make_movie(source, sequence, OUTDIR, window_kwargs=window_kwargs)
        # os.remove(os.path.join(OUTDIR, '00000000.tga'))

    make_movie(imgname=OUTDIR+'/', movname=OUTDIR + '_movie', indexsz='08', ext='tga', framerate=10)
