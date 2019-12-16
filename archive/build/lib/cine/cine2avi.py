#!/usr/bin/env python
from . import cine
from numpy import *
import os, sys
import argparse


parser = argparse.ArgumentParser(description="Convert CINE file(s) to an AVI.  Also works on TIFFs.")
parser.add_argument('cines', metavar='cines', type=str, nargs='+', help='input cine file(s), append [start:end:skip] (python slice notation) to filename to convert only a section')
parser.add_argument('-o', dest='output', type=str, default='%s.avi', help='output filename, may use %%s for input filename w/o extension or %%d for input file number')
parser.add_argument('-g', dest='gamma', type=float, default=1., help='gamma of output, assumes input is gamma 1, or: I -> I**(1/gamma); use 2.2 to turn linear image to "natural" for display [default: 1]')
parser.add_argument('-f', dest='framerate', type=int, default=30, help='frames per second [default: 30]')
parser.add_argument('-q', dest='quality', type=int, default=75, help='JPEG quality (0-100) [default: 75]')
parser.add_argument('-c', dest='clip', type=float, default=0, help='histogram clip, clip the specified fraction to pure black and pure white, and interpolate the rest; applied before gamma; recommended is 1E-4 - 1E-3 [default: 0]')
parser.add_argument('-s', dest='hist_skip', type=int, default=10, help='only check every Nth frame when computing histogram clip [default: 5]')
parser.add_argument('-r', dest='rotate', type=int, default=0, help='amount to rotate in counterclockwise direction, must be multiple on 90 [default: 0]')
args = parser.parse_args()

def noneint(s):
    return None if not s else int(s)

for i, fn in enumerate(args.cines):
    fn = fn.strip()
    
    frame_slice = slice(None)
    if '[' in fn:
        if fn[-1] == ']':
            fn, s = fn.split('[')
            try:
                frame_slice = slice(*list(map(noneint, s[:-1].split(':'))))
            except:
                raise ValueError("Couldn't convert '[%s' to slice notation" % s)

        else:
            print("Warning, found '[' in input, but it didn't end with ']', so I'll assume you didn't mean to give a frame range.")
    
    base, ext = os.path.splitext(fn)
    ext = ext.lower()
    
    if not os.path.exists(fn):
        print("File %s not found, ignoring." % fn)
        continue
    
    output = args.output
    if '%s' in args.output: output = output % base
    elif '%' in args.output: output = output % i
    
    bpp = None
    
    if ext in ('.cin', '.cine'):
        input = cine.Cine(fn)
        bpp = input.real_bpp
        if bpp < 8 or bpp > 16: bpp = None #Just in case
        
    elif ext in ('.tif', '.tiff'):
        input = cine.Tiff(fn)
        
    bpps = input[0].dtype.itemsize * 8
    if bpp is None: bpp = bpps
    
    frames = list(range(*frame_slice.indices(len(input))))
    
    if args.clip == 0:
        map = linspace(0., 2.**(bpps - bpp), 2**bpps)
    else:
        counts = 0
        bins = arange(2**bpps + 1)
        
        for i in frames[::args.hist_skip]:
            c, b = histogram(input[i], bins)
            counts += c
        
        counts = counts.astype('d') / counts.sum()
        counts = counts.cumsum()
        
        bottom_clip = where(counts > args.clip)[0]
        if not len(bottom_clip): bottom_clip = 0
        else: bottom_clip = bottom_clip[0]

        top_clip = where(counts < (1 - args.clip))[0]
        if not len(top_clip): top_clip = 2**bpps
        else: top_clip = top_clip[-1]

        #print bottom_clip, top_clip
        #import pylab
        #pylab.plot(counts)
        #pylab.show()
        #sys.exit()

        m = 1. / (top_clip - bottom_clip)
        map = clip(-m * bottom_clip + m * arange(2**bpps, dtype='f'), 0, 1)
            
    map = map ** (1./args.gamma)
    
    map = clip(map * 255, 0, 255).astype('u1')

    print('%s -> %s' % (fn, output))
    
    output = cine.Avi(output, framerate=args.framerate, quality=args.quality)
    
    #print frames
    for i in frames:
        frame = input[i]
        if args.rotate:
            frame = rot90(frame, (args.rotate%360)//90)
        output.add_frame(map[frame])
        
    output.close()
        
    