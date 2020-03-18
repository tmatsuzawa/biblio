#!/usr/bin/env python
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import shaders
from numpy import *
from scipy.interpolate import interp1d
import sys, os, shutil
import time
import argparse
from PIL import Image
import traceback
import json
import re
#import mesh
import argparse
import glob
import io
ESCAPE = b'\x1b'

#XR = 0
#YR = 0
#ZR = 0
    

mag  = lambda X: sqrt((asarray(X)**2).sum(-1))
mag1 = lambda X: sqrt((asarray(X)**2).sum(-1))[..., newaxis]
dot  = lambda X, Y: (asarray(X)*Y).sum(-1)
dot1 = lambda X, Y: (asarray(X)*Y).sum(-1)[..., newaxis]
norm = lambda X: X / mag1(X)
rms = lambda X: sqrt((X**2).sum() / len(X))
plus  = lambda X: roll(X, -1, 0)
minus = lambda X: roll(X, +1, 0)


NORM_ERROR = False
if NORM_ERROR:
    def norm(X):
        if (mag(X) == 0).any():
            raise ValueError('Tried to normalize magnitude 0 vector!')
        else:
            return X / mag1(X)


def proj(X, Y):
    Yp = norm(Y)
    return dot(Yp, X) * Yp


def gram_schmidt(V):
    V = list(array(v, dtype='d') for v in V)
    U = []
    for v in V:
        v2 = v.copy()
        for u in U: v -= dot(v2, u) * u
        U.append(norm(v))
    return array(U)
    
    
def normalize_basis(V):
    V = gram_schmidt(V[:2])
    V = vstack((V, cross(V[0], V[1])))
    return V


class ListValue:
    def __init__(self, list, value=0):
        self.list = list
        self.value = value
    
    def val(self):
        self.value = self.value % len(self.list)
        return self.list[self.value]
        
    def adj(self, i=1):
        self.value = int(self.value + i) % len(self.list)

    def set(self, value):
        self.value = int(value) % len(self.list)


class LinValue:
    def __init__(self, value=0, change=1., cycle=None):
        self.value = value
        self.change = change
        self.cycle = cycle

    def val(self):
        return self.value
    
    def adj(self, i=1):
        if i is None:
            self.set(0)
        else:
            self.set(self.value + i * self.change)
                
    def set(self, value):
        self.value = value
        
        if self.cycle is not None:
            self.value %= self.cycle        
        

class LogValue(LinValue):
    def adj(self, i=1):
        self.set (self.value * self.change ** i)
        
        
def rotate_by_vector(X, r):
    if mag(r) < 1E-10: return X
    
    zn = norm(r)
    phi = mag(r)
    xn = eye(3)[argmin(abs(zn))]
    xn = norm(xn - zn * dot(xn, zn))
    yn = cross(zn, xn)

    x = dot1(X, xn)
    y = dot1(X, yn)
    z = dot1(X, zn)
    
    xr = x * cos(phi) - y * sin(phi)
    yr = x * sin(phi) + y * cos(phi)
    
    return xr * xn + yr * yn + z*zn
    
        
class RotMatrix:
    def __init__(self, value=None):
        if value is None: value = eye(3)
        self.value = value
        
    def val(self):
        return self.value
    
    def adj(self, r=(1, 0, 0)):
        if r is None:
            self.value = eye(3)
        
        elif mag(r) > 1E-6:                    
            self.set(rotate_by_vector(self.value, r))

    def normalize(self):
        self.value = normalize_basis(self.value)
        
    def set(self, value):
        self.value = normalize_basis(value)
    
class BoolValue(LinValue):
    def __init__(self, value):
        self.value = value
        
    def adj(self, i=1):
        if i: self.value = not self.value

    def set(self, value):
        self.value = bool(value)

LAST_TIME = None
FULLSCREEN = None

MOUSE_CAPTURED = False
MOUSE_X = 0
MOUSE_Y = 0


ROTATE_AMOUNT = 45/4. * pi / 180
SCREENSHOT = None
BASENAME = '' #os.path.splitext(sys.argv[0])[0]

XN, YN, ZN = eye(3)

COLORS = [
    None, #Default coloration
    ( 1.        ,  0.        ,  0.        ), #Red
    ( 1.        ,  0.50196078,  0.        ), #Orange
    ( 0.62745098,  0.62745098,  0.        ), #Yellow
    ( 0.        ,  0.62745098,  0.        ), #Green
    ( 0.        ,  0.62745098,  0.62745098), #Teal
    ( 0.        ,  0.        ,  1.        ), #Blue
    ( 1.        ,  0.        ,  1.        ), #Magenta
    ( 1.        ,  0.62745098,  0.62745098), #Pink
    ( 1.        ,  1.        ,  1.        ), #White
]

class View:
    def __init__(self, **kwargs):
        self.params = {
            'R':             RotMatrix(kwargs.get('R', None)),
            'frame_rate':    LogValue(kwargs.get('frame_rate', 15.), 2**.5),
            'display_3d':    BoolValue(kwargs.get('display_3d', False)),
            'eye_split':     LogValue(kwargs.get('eye_split', 0.02), 2**.25),
            'frame':         LinValue(kwargs.get('frame', 0.0)),
            'playing':       BoolValue(kwargs.get('playing', False)),
#            'fullscreen':    BoolValue(kwargs.get('fullscreen', False)), // This should be global
            'draw_box':      BoolValue(kwargs.get('draw_box', True)),
            'autorotate':    BoolValue(kwargs.get('autorotate', False)),
            'autorotate_speed': LogValue(kwargs.get('autorotate_speed', 0.4*pi), 2**.25),
            'background_color': LinValue(kwargs.get('background_color', 1), 1, 3),
            'z_shift':       LogValue(kwargs.get('z_shift', 2), 2**.25),
            'fov':           LogValue(kwargs.get('fov', 45), 2**.25),
            'perspective':   BoolValue(kwargs.get('perspective', True)),
            'brightness':    LogValue(kwargs.get('brightness', 1.0), 2**.125),
            'display_frame': BoolValue(kwargs.get('display_frame', True)),
            'multi_frame':   LinValue(kwargs.get('multi_frame', 1), 1, 30),
            'multi_frame_skip':LinValue(kwargs.get('multi_frame_skip', 1), 1, 30),
            'tile':          LinValue(kwargs.get('tile', 0), 1, 3),
            'tile_x':        LinValue(kwargs.get('tile_x', 0), 0),
            'tile_y':        LinValue(kwargs.get('tile_y', 0), 0),
            'tile_z':        LinValue(kwargs.get('tile_z', 0), 0),
            'render_type':   LinValue(kwargs.get('render_type', 1), 1, 4),
            'colorize':      LinValue(kwargs.get('colorize', 0), 1, len(COLORS)),
            'box_x0':        LinValue(kwargs.get('box_x0', 0), 0),
            'box_x1':        LinValue(kwargs.get('box_x1', 0), 0),
            'box_y0':        LinValue(kwargs.get('box_y0', 0), 0),
            'box_y1':        LinValue(kwargs.get('box_y1', 0), 0),
            'box_z0':        LinValue(kwargs.get('box_z0', 0), 0),
            'box_z1':        LinValue(kwargs.get('box_z1', 0), 0),
            'box_linewidth': LinValue(kwargs.get('box_linewidth', 2), 1, 5),
            'ZU':            LinValue(kwargs.get('ZU',  0.45), 0.005),
            'ZD':            LinValue(kwargs.get('ZD', -0.45), 0.005),
            'YU':            LinValue(kwargs.get('YU',  0.45), 0.005),
            'YD':            LinValue(kwargs.get('YD', -0.45), 0.005),
            'XU':            LinValue(kwargs.get('XU',  0.45), 0.005),
            'XD':            LinValue(kwargs.get('XD', -0.45), 0.005),
        }

        self.key_presses = {
            '<': ('frame_rate', -1), ',': ('frame_rate', -1),
            '>': ('frame_rate', +1), '.': ('frame_rate', +1),
            ' ': 'playing',
            '3': 'display_3d',
            '[': ('eye_split', -1),  '{': ('eye_split', -1), 
            ']': ('eye_split', +1),  '}': ('eye_split', +1),
            'w': ('R', -ROTATE_AMOUNT * XN),
            's': ('R', +ROTATE_AMOUNT * XN),
            'a': ('R', -ROTATE_AMOUNT * YN),
            'd': ('R', +ROTATE_AMOUNT * YN),
            'e': ('R', -ROTATE_AMOUNT * ZN),
            'q': ('R', +ROTATE_AMOUNT * ZN),
            'Q': ('ZU', +1),
            'A': ('ZU', -1),
            'W': ('ZD', +1),
            'S': ('ZD', -1),
            'E': ('YU', +1),
            'D': ('YU', -1),
            'R': ('YD', +1),
            'F': ('YD', -1),
            'T': ('XU', +1),
            'G': ('XU', -1),
            'Y': ('XD', +1),
            'H': ('XD', -1),
            'b': 'draw_box',
            'n': 'background_color',
            'c': 'render_type',
            'z': ('z_shift', -1),
            'x': ('z_shift', +1),
            'i': ('R', None),
            'o': ('frame', None),
            'r': 'autorotate',
            't': ('autorotate_speed', -1),
            'y': ('autorotate_speed', +1),
            '9': ('colorize', -1),
            '0': ('colorize', +1),
            GLUT_KEY_LEFT:  ('frame', -1),
            GLUT_KEY_RIGHT: ('frame', +1),
            GLUT_KEY_UP:    ('frame', +10),
            GLUT_KEY_DOWN:  ('frame', -10),
            '_': ('brightness', -1), '-': ('brightness', -1),
            '+': ('brightness', +1), '=': ('brightness', +1),
            'f': 'display_frame',
            'k': ('multi_frame', -1),
            'l': ('multi_frame', +1),
            ';': ('multi_frame_skip', -1),
            "'": ('multi_frame_skip', +1),
            'g': ('fov', -1),
            'h': ('fov', +1),
            '?': 'tile',
            '/': 'tile',
            'p': 'perspective',
        }
        
        self.smooth_params = ['R', 'frame', 'z_shift', 'fov']
        self.hidden_params = ['playing', 'autorotate', 'autorotate_speed']
        
        self.changed = True
        self.frames = []
        self.labels = []
        
    def copy(self):
        return View(**self.param_dict())
        
    def param_dict(self, format=False):
        values = {}
        
        for k, v in self.params.items():
            v = v.value
            if format:
                if type(v) is ndarray and v.shape == (3, 3):
                    v1, v2, v3 = map(lambda x: '(%+6.3f, %+6.3f, %+6.3f)' % tuple(x), v) 
                    values[k] = '[%s, %s]' % (v1, v2)
                elif k == 'frame':
                    values[k] = str(int(v))
                else:
                    values[k] = str(v)

            else:
                values[k] = v

        return values     
        
    def append(self, frame, label=''):
        self.frames.append(frame)
        self.labels.append(label)
        self.params['frame'].cycle = len(self.frames)

    def __getattr__(self, name):
        if name in self.params:
            return self.params[name].val()
        else:
            raise AttributeError("'View' object has no attribute '%s'" % name)
    
    def key_press(self, key):
        if type(key) is bytes:
            key = key.decode('ascii')
        if key in self.key_presses:
            v = self.key_presses[key]
            if type(v) in (tuple, list):
                self.params[v[0]].adj(v[1])
            else:
                self.params[v].adj()
        
            self.changed = True
            
    def adj(self, var, inc):
        self.params[var].adj(inc)
        self.changed = True
        
    def update(self, d):
        for k, v in d.items():
            self.params[k].set(v)
    
VIEW = View()
    
    

def if_int(str):
    if not str: return None
    else: return int(str)
    
    
THICKNESS_RATIO = 1./10
    
def show(input_files, movie_views=None, movie_image_name=None, window_name='4d viewer', scale=None, offset=None, thickness=None, window_kwargs={}):
    global WINDOW, VIEW
    
    for f in input_files:
        bn = os.path.basename(os.path.splitext(f)[0])
        ext = os.path.splitext(f)[1].lower()
        if ext == '.ply':
            VIEW.append(Mesh(f, colors=(1, .25, .25)), label=bn)
        elif ext == '.pickle':
            VIEW.append(pickle.load(open(f, 'rb')), label=bn)
        elif ext in ('.json', '.tangle', '.path'):
            traces, info = json_to_traces(f)
            
            #Accept non plural versions
            for attr in ['normal', 'color']:
                pa = attr + 's'
                if attr in info and pa not in info:
                    info[pa] = info[attr]
            
            m = Mesh()
            closed = info.get('closed', True)
            if not hasattr(closed, '__iter__'):
                closed = [closed] * len(traces)
            
            if 'thickness' in info:
                ts = info['thickness']
                if type(ts) not in (list, tuple):
                    ts = [ts] * len(traces)
            else:
                if thickness is not None: t = thickness
                else:
                    X = vstack(traces)
                    t = sqrt(((X - X.mean(0))**2).sum(-1).mean(0)) * THICKNESS_RATIO
                    
                ts = [t] * len(traces)
                         
            if 'normals' in info:
                for trace, n, t, cl in zip(traces, info['normals'], ts, closed):
                    n = array(n)
                    #m += trace_ribbon(trace, n, width=t, colors=(1, .25, .25))
                    m += trace_split_ribbon(trace, n, width=t, closed=cl)
                    #m += trace_ribbon(trace, n, width=t, colors=(1, 1, 1)) #colors=trace[:, 2],  cmap='Spectral')
                    #m += trace_ribbon(trace, n, width=t, colors=trace[:, 2],  cmap='spectral')
            else:
                if 'colors' in info:
                    #colors = array(info[colors])
                    colors = info['colors']
                    if type(colors[0]) in (int, tuple):
                        if len(colors) in (3, 4):
                            colors = [colors] * len(traces)
                        else:
                            raise ValueError('colors field should have 3 or 4 elements (RGB or RGBA)')
                    else:
                        colors = [array(color) for color in colors]
                        if colors[0].ndim not in (1, 2):
                            raise ValueError('colors field in tangle file should be a single color (all paths the same), a list of colors (a different color for each path), or an list of arrays of colors (a different color for each point on each array)')
                        elif len(colors) != len(traces):
                            raise ValueError('length of colors field should match length of paths field')
                        elif colors[0].shape[-1] not in (3, 4):
                            raise ValueError('colors field should have 3 or 4 elements (RGB or RGBA)')
                else:
                    colors = [(1, .25, .25)] * len(traces)
                
                for trace, t, c, cl in zip(traces, ts, colors, closed):
                    #m += trace_line(trace, thickness=t, colors=trace[:, 2],  cmap='spectral')
                    m += trace_line(trace, thickness=t, colors=c, closed=cl)
                    
            VIEW.append(m, label=bn)
            
        else:
            if ',' in open(f).readline():
                trace = loadtxt(f, delimiter=',')
            else:
                trace = loadtxt(f)
            
            if thickness is not None: t = thickness
            else:
                t = sqrt(((trace - trace.mean(0))**2).sum(-1).mean(0)) * THICKNESS_RATIO

            VIEW.append(trace_line(trace, thickness=thickness), label=bn)

        
    all_points = vstack([m.points for m in VIEW.frames])
    mi = all_points.min(0) 
    ma = all_points.max(0)
        
    if offset is None:
        offset = (ma + mi)/2.
        #offset = mesh_list[0].points.mean(0)
        
    mi -= offset
    ma -= offset
    
    for m in VIEW.frames: m.points -= offset
    
    if scale is None:
        #scale = mean(sqrt((mesh_list[0].points ** 2).sum(-1)))
        scale = (ma - mi).max()
    mi /= scale
    ma /= scale
    
    VIEW.params['tile_x'].set(ma[0] - mi[0])
    VIEW.params['tile_y'].set(ma[1] - mi[1])
    VIEW.params['tile_z'].set(ma[2] - mi[2])

    #b0 = mi.min()
    #b1 = ma.max()
    b0 = -0.5
    b1 = 0.5

    VIEW.params['box_x0'].set(b0)
    VIEW.params['box_x1'].set(b1)
    VIEW.params['box_y0'].set(b0)
    VIEW.params['box_y1'].set(b1)
    VIEW.params['box_z0'].set(b0)
    VIEW.params['box_z1'].set(b1)
    
    for m in VIEW.frames: m.points /= scale  
    
    glutInit(sys.argv)

    
    WINDOW = GLUTWindow(window_name, **window_kwargs)
    WINDOW.draw_func = WINDOW.draw_scene
    WINDOW.idle_func = WINDOW.draw_scene
    
    if movie_views is None:
        WINDOW.keyboard_func = key_pressed
        WINDOW.special_func = special_pressed
        WINDOW.mouse_func = mouse_func
        WINDOW.motion_func = motion_func
        
        print('''----Keys----
wasdqe -> Rotate volume
zx -> Zoom
i -> Reset rotation
o -> Reset frame
gh -> Adjust field of view
arrows -> Skip forward/backward
space -> Pause/play
+- -> Adjust brightness
<> -> Adjust playback speed
ESC -> exit
3 -> Activate/deactivate stereo anaglyph
[] -> Adjust eye distance
Tab -> Toggle fullscreen
Left mouse/Drag -> Rotate
r -> Autorotate
ty -> Adjust autorotate speed
1 -> Take screenshot
m -> Create/add entry to movie file
f -> Show/hide frame number
kl -> Adjust multi-frame view number
;' -> Adjust multi-frame view skip
''')

    else:
        WINDOW.keyboard_func = key_pressed_exit_only
        WINDOW.movie_views = movie_views
        img_name = movie_image_name
        if img_name is None: img_name = '%08d.tga'
        WINDOW.image_name = img_name
        WINDOW.frame_num = 0

    WINDOW.start() 

def switch_lighting(lighting_type, brightness=0.75):
    glLoadIdentity()
    B = array([brightness])
    #if mono: B*= 2
    spec_B = B * 0.3
    
    glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)    
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (1, 1.0, 1.0, 1))
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (1, 1, 1, 1))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 80.0)

    if lighting_type == 0:
        glLightfv(GL_LIGHT0, GL_DIFFUSE, B * (1, 1, 1, 1))        
        glLightfv(GL_LIGHT0, GL_SPECULAR, (0, 0, 0, 0))
#        glLightfv(GL_LIGHT0, GL_SPECULAR, B * (1, 1, 1, 1))
        glLightfv(GL_LIGHT0, GL_POSITION, 10*norm((0, 1, -.5, 0)))
        glEnable(GL_LIGHT0)
    
        glLightfv(GL_LIGHT1, GL_DIFFUSE, B * (1, 1, 1, 1))        
#        glLightfv(GL_LIGHT1, GL_SPECULAR, B * (1, 1, 1, 1))
        glLightfv(GL_LIGHT1, GL_SPECULAR, (0, 0, 0, 0))
        glLightfv(GL_LIGHT1, GL_POSITION, 10*norm((-1, -1, -.5, 0)))
        glEnable(GL_LIGHT1)
    
        glLightfv(GL_LIGHT2, GL_DIFFUSE, B * (1, 1, 1, 1))        
        glLightfv(GL_LIGHT2, GL_SPECULAR, (0, 0, 0, 0))
#        glLightfv(GL_LIGHT2, GL_SPECULAR, B * (1, 1, 1, 1))
        glLightfv(GL_LIGHT2, GL_POSITION, 10*norm((1, -1, -.5, 0)))
        glEnable(GL_LIGHT2)

    elif lighting_type == 1:
#        C = array([0.0, 0.125, 0.25, 1])
#        C = array([0.125, 0.25, 0.5, 1])
        C = array([0.2, 0.25, 0.3, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, B * C)
        glLightfv(GL_LIGHT0, GL_SPECULAR, 0*(1+C))
        glLightfv(GL_LIGHT0, GL_POSITION, (.5, -1, .25, 0))
        glEnable(GL_LIGHT0)
        
#        C = array([0.8, 0.6, 0.4, 1])
        C = array([0.7, 0.65, 0.6, 1])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, B * C)
        glLightfv(GL_LIGHT1, GL_SPECULAR, spec_B*(1+C))
        glLightfv(GL_LIGHT1, GL_POSITION, (-1, 1, 2, 0))
        glEnable(GL_LIGHT1)
    
#        C = array([0, 0.125, .05, 1])
#        C = array([0.25, 0.5, .375, 1])
        C = array([0.4, 0.4, .4, 1])
        glLightfv(GL_LIGHT2, GL_DIFFUSE, B * C)
        glLightfv(GL_LIGHT2, GL_SPECULAR, 0*(1+C))
        glLightfv(GL_LIGHT2, GL_POSITION, (-.5, -1, .25, 0))
        glEnable(GL_LIGHT2)


    elif lighting_type == 2:
        C = array([0.25, 0, 1, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, B * C)
        glLightfv(GL_LIGHT0, GL_SPECULAR, spec_B*(1+C))
        glLightfv(GL_LIGHT0, GL_POSITION, (1, -1, .25, 0))
        glEnable(GL_LIGHT0)
        
        C = array([0.8, 0.25, 0, 1])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, B * C)
        glLightfv(GL_LIGHT1, GL_SPECULAR, spec_B*(1+C))
        glLightfv(GL_LIGHT1, GL_POSITION, (-1, 1, .75, 0))
        glEnable(GL_LIGHT1)
    
        C = array([0, 0.8, 0, 1])
        glLightfv(GL_LIGHT2, GL_DIFFUSE, B * C)
        glLightfv(GL_LIGHT2, GL_SPECULAR, spec_B*(1+C))
        glLightfv(GL_LIGHT2, GL_POSITION, (1.2, 1, .75, 0))
        glEnable(GL_LIGHT2)


def rot_x(a, x=eye(3)):
    x = array(x)
    rx = x.copy()
    rx[..., 1] = cos(a) * x[..., 1] - sin(a) * x[..., 2]
    rx[..., 2] = cos(a) * x[..., 2] + sin(a) * x[..., 1]
    return rx
    
def rot_y(a, x=eye(3)):
    x = array(x)
    rx = x.copy()
    rx[..., 2] = cos(a) * x[..., 2] - sin(a) * x[..., 0]
    rx[..., 0] = cos(a) * x[..., 0] + sin(a) * x[..., 2]
    return rx
 
def rot_z(a, x=eye(3)):
    x = array(x)
    rx = x.copy()
    rx[..., 0] = cos(a) * x[..., 0] - sin(a) * x[..., 1]
    rx[..., 1] = cos(a) * x[..., 1] + sin(a) * x[..., 0]
    return rx

def rot_xyz(ax, ay, az, x=eye(3)):
    return gram_schmidt(rot_z(az, rot_y(ay, rot_x(ax, x))))


UNIT_BOX = array([(x, y, z) for x in (1, -1) for y in (1, -1) for z in (1, -1)])

#TICK_TIME = None
#
#def tick():
#    global TICK_TIME
#    TICK_TIME = time.time()
#    
#def tock(message=''):
#    global TICK_TIME
#
#    if TICK_TIME is None: return
#
#    now = time.time()
#    print message + '%.3f' % (now - TICK_TIME)
#    TICK_TIME = now
#    
        
class FuncSetter(object):
    def __init__(self, prop_name, glut_func=None):
        self.prop_name = prop_name
        self.glut_func = glut_func
        
    def __get__(self, obj, obj_type):
        return getattr(obj, '__' + self.prop_name, None)
        
    def __set__(self, obj, val):
        try:
            if self.glut_func is not None: self.glut_func(val)
            else: print('Warning: {:s} property is ignored.'.format(self.prop_name))
        except:
            print("Failed to set callback for {:s}".format(self.prop_name))
        






#modification of OpenGL.GL.shaders.compileProgram, which does not allow for setting variables before validation.
def compileProgram(*s, **vars):
    program = glCreateProgram()
    
    for ss in s:
        glAttachShader(program, ss)
        
    glLinkProgram(program)
        
    shaders.glUseProgram(program)
    set_uniforms(program, **vars)
    shaders.glUseProgram(0)
        
    glValidateProgram(program)
    
    validation = glGetProgramiv(program, GL_VALIDATE_STATUS)
    if validation == GL_FALSE:
        raise RuntimeError(
            """Validation failure (%s): %s"""%(
            validation,
            glGetProgramInfoLog(program),
        ))
    
    link_status = glGetProgramiv(program, GL_LINK_STATUS)
    if link_status == GL_FALSE:
        raise RuntimeError(
            """Link failure (%s): %s"""%(
            link_status,
            glGetProgramInfoLog(program),
        ))
    
    for ss in s:
        glDeleteShader(ss)
        
        
        
    return shaders.ShaderProgram(program)


def set_uniforms(program, **vars):
    
    for key, val in vars.items():
        val = asarray(val)
        if not val.shape: val.shape = (1,)
        
        if len(val) > 4: raise ValueError('at most 4 values can be used for set_uniforms')
        if val.dtype in ('u1', 'u2', 'u4', 'u8', 'i1', 'i2', 'i4', 'i8'): dt = int
        elif val.dtype in ('f', 'd'): dt = float
        else: raise ValueError('values for set_uniforms should be ints or floats')

        #print key, dt, len(val)
        glUniforms[dt, len(val)](glGetUniformLocation(program, key), *val)


UNIT_BOX = array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype='f')
BOX_EDGES = array([(i, j) for i in range(8) for j in range(8) if ((i < j) and sum((UNIT_BOX[i] - UNIT_BOX[j])**2) == 1.)])
BOX_FACES = array([
    (0, 1, 3, 2), #-X
    (4, 6, 7, 5), #+X
    (0, 4, 5, 1), #-Y
    (2, 3, 7, 6), #+Y
    (0, 2, 6, 4), #-Z
    (1, 5, 7, 3), #+Z
], dtype='u4')

BOX_EDGE_CENTERS = 0.5 * (UNIT_BOX[BOX_EDGES[:, 0]] + UNIT_BOX[BOX_EDGES[:, 1]])
        
class GLUTWindow(object):
    draw_func = FuncSetter('draw_func', glutDisplayFunc)
    idle_func = FuncSetter('idle_func', glutIdleFunc)
    keyboard_func = FuncSetter('keyboard_func', glutKeyboardFunc)
    special_func = FuncSetter('special_func', glutSpecialFunc)
    motion_func = FuncSetter('motion_func', glutMotionFunc)
    passive_motion_func = FuncSetter('passive_motion_func', glutPassiveMotionFunc)
    mouse_func = FuncSetter('mouse_func', glutMouseFunc)
    
    def __init__(self, window_name, width=1000, height=1000, fov=45, depth_test=False, min_z=0.01, max_z=10., ortho_height=1.):
        self.width = width
        self.height = height
        self.fov = fov
        self.depth_test = depth_test
        self.min_z = min_z
        self.max_z = max_z
        self.ortho_height = ortho_height

#        glutInit(sys.argv)

        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE)
        
        glutInitWindowSize(width, height)
        
        self.window = glutCreateWindow(window_name)
        glEnable(GL_MULTISAMPLE)
        
        glutReshapeFunc(self.resize_GL)


        self.shiny_vertex_shader = shaders.compileShader('''
            varying vec3 normal;
            varying vec4 color;
                                                         
            void main()
            {
                normal = normalize(gl_NormalMatrix * gl_Normal);
                color = gl_Color;
                gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;
            }
        ''', GL_VERTEX_SHADER)

        
        self.shiny_fragment_shader = shaders.compileShader('''
            varying vec3 normal;                                               
            varying vec4 color;                                    
                                                           
            void main()
            {
                vec4 out_color = vec4(0.0, 0.0, 0.0, 1.0);
                vec3 n = normal;
                float ndl;
                
                //if (n.z < 0.0) {n = -1.0 * n;}
                if (!gl_FrontFacing) {n = -1.0 * n;}
                
                for (int i = 0; i < 3; i ++) {
                    ndl = dot(normalize(n), normalize(vec3(gl_LightSource[i].position)));
                    
                    if (ndl > 0.0) {
                        out_color += color * gl_LightSource[i].diffuse * ndl;
                        out_color += gl_LightSource[i].specular * pow(ndl, gl_FrontMaterial.shininess);
                    }
                }
                
                /*
                if (gl_FrontFacing) {
                    color = color * vec4(1.0, 1.0, 0.8, 1.0);                    
                } else {                    
                    color = color * vec4(0.8, 0.8, 1.8, 1.0);
                }
                */
                
                gl_FragColor = out_color;
            }
        ''', GL_FRAGMENT_SHADER)


        self.skyline_vertex_shader = shaders.compileShader('''
            varying vec3 normal;
            varying vec4 color;                                    
                                                         
            void main()
            {
                normal = normalize(gl_NormalMatrix * gl_Normal);
                color = gl_Color;
                gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;
            }
        ''', GL_VERTEX_SHADER)
        
        self.skyline_fragment_shader = shaders.compileShader('''
            varying vec3 normal;
            varying vec4 color;    
            #define LINE_WIDTH 0.2
                       
                                                           
            void main()
            {
                vec4 out_color = vec4(0.0, 0.0, 0.0, 1.0);
                vec3 n = normal;
                float ndl;
                
                if (!gl_FrontFacing) {n = -1.0 * n;}
                
                for (int i = 0; i < 3; i ++) {
                    ndl = dot(normalize(n), normalize(vec3(gl_LightSource[i].position)));
                    
                    if (ndl > 0.0) {
                        out_color += color * gl_LightSource[i].diffuse * ndl * gl_FrontMaterial.diffuse;
                        out_color += gl_LightSource[i].specular * pow(ndl, gl_FrontMaterial.shininess)* gl_FrontMaterial.specular;
                    }
                }
                
                if (n.y > -LINE_WIDTH) {
//                    color += sqrt(1.0-abs(n.z)) * vec4(0.5, 0.5, 1.0, 0.0) * clamp(LINE_WIDTH + n.y / (2.0*LINE_WIDTH), 0.0, 1.0);
//                    out_color += sqrt(1.0-abs(n.z)) * vec4(0.25, 0.25, 0.35, 0.0) * cos(1.5707963267948966*(1.0-clamp(LINE_WIDTH + n.y / (2.0*LINE_WIDTH), 0.0, 1.0)));
                    out_color += sqrt(1.0-abs(n.z)) * vec4(0.5, 0.5, 0.7, 0.0) * cos(1.5707963267948966*(1.0-clamp(LINE_WIDTH + n.y / (2.0*LINE_WIDTH), 0.0, 1.0)));
                }
                
                gl_FragColor = out_color;
            }
        ''', GL_FRAGMENT_SHADER)


        self.simple_vertex_shader = shaders.compileShader('''
            varying vec3 normal;
            varying vec4 color;
                                                         
            void main()
            {
                normal = normalize(gl_NormalMatrix * gl_Normal);
                color = gl_Color;
                gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;
            }
        ''', GL_VERTEX_SHADER)

        
        self.simple_fragment_shader = shaders.compileShader('''
            varying vec3 normal;                                               
            varying vec4 color;                                    
                                                           
            void main()
            {
                float b = clamp(sqrt(normal.z), 0.0, 1.0);
            
                gl_FragColor = color * vec4(b, b, b, 1.0);
            }
        ''', GL_FRAGMENT_SHADER)
    
        self.shiny_shader = compileProgram(self.shiny_vertex_shader, self.shiny_fragment_shader)
        self.skyline_shader = compileProgram(self.skyline_vertex_shader, self.skyline_fragment_shader)
        self.simple_shader = compileProgram(self.simple_vertex_shader, self.simple_fragment_shader)
 
        self.render_shaders = [self.shiny_shader, self.skyline_shader, self.shiny_shader, self.simple_shader]
 
        self.last_time = None
        
        
    def init_GL(self):
        glClearColor(0.0, 0.0, 0.0, 0.0) 

        if self.depth_test:
            glClearDepth(1.0) 
            glDepthFunc(GL_LESS) 
            glEnable(GL_DEPTH_TEST)

        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        #glEnable(GL_BLEND)
        #glBlendFunc(GL_ONE, GL_ONE)
        glShadeModel(GL_SMOOTH) 
            
        self.resize_GL(self.width, self.height)
        

    def resize_GL(self, width=None, height=None):
        global VIEW

        if width is None: width = self.width
        if height is None: height = self.height
        if height == 0: height = 1
        self.width = width
        self.height = height
    
        glViewport(0, 0, width, height) 
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        aspect_ratio = width/float(height)
        if self.fov:
            gluPerspective(self.fov, aspect_ratio, self.min_z, self.max_z)
        else:
            a = 0.5 * self.ortho_height
            glOrtho(-a * aspect_ratio, a * aspect_ratio, -a, a, self.min_z, self.max_z)
            
        glMatrixMode(GL_MODELVIEW)
        VIEW.changed = True


    def fullscreen(self, fullscreen=False):
        if fullscreen:
            self.pre_height = self.height
            self.pre_width = self.width
            glutFullScreen()
        else:
            glutReshapeWindow(self.pre_width, self.pre_height)
            

    def start(self):
        self.init_GL()

        glutMainLoop()
        
    def screenshot(self, fn=None, A=True, fix_bg=True):
        if A:
            channels = "RGBA"
            glc = GL_RGBA
        else:
            channels = "RGB"
            glc = GL_RGB
        
        glFlush(); glFinish() #Make sure drawing is complete
        
        glReadBuffer(GL_FRONT)
        img = glReadPixels(0, 0, self.width, self.height, glc, GL_UNSIGNED_BYTE)
        #print len(img), self.width, self.height, len(channels)
        img = frombuffer(img, dtype='u1').reshape((self.height, self.width, len(channels)))

        glFlush(); glFinish() #Make sure drawing is complete

        
        if A:
            img = img.copy()


            if fix_bg:
                trans = img[..., 3] == 0
                
                if trans.any():
                    bg = img[where(trans)][..., :3].mean(0) #Mean color of all transparent pixels
                    print(bg)
                    nt = where((img[..., 3] != 0) & (img[..., 3] != 255))
                    
                    rgb = img[nt][..., 0:3]
                    a = img[nt][..., 3:4]/255.
                    rgb = clip((rgb - bg * (1-a)) / a, 0, 255).astype('u1')
                    
                    img[nt][..., :3] = rgb

            else:
                img[..., 3] = (img[..., 3] > 0) * 255
        
        img = Image.fromarray(img[::-1])

        #img = Image.frombuffer(channels, (self.width, self.height), img, "raw", channels, 0, 0)
        if fn:
            img.save(fn)
            
        return img
    
    
    def draw_scene(self, view=None):
        global VIEW, SCREENSHOT, MOUSE_CAPTURED, COLORS
        
        if view is None:
            view = VIEW
            
        changed = view.changed
        cf = int(view.frame+1E-3) % len(view.frames)

        if self.last_time is None: self.last_time = time.time()
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.last_time = current_time
        
        if hasattr(self, 'movie_views'):
            if self.frame_num >= len(self.movie_views):
                sys.exit()   
            
            view.update(self.movie_views[self.frame_num])
            changed = True
            cf = int(view.frame+1E-3) % len(view.frames)
            SCREENSHOT = self.image_name % self.frame_num
            self.frame_num += 1

            fov = view.fov if view.perspective else 0            
            min_z = view.z_shift * 0.1
            max_z = view.z_shift * 10
            ortho_height = 2. * view.z_shift * tan(view.fov * pi / 360)
        
            if fov != self.fov or min_z != self.min_z or max_z != self.max_z or ortho_height != self.ortho_height:
                self.fov = fov
                self.min_z = min_z
                self.max_z = max_z
                self.ortho_height = ortho_height
                self.resize_GL()

        else:
        
            fov = view.fov if view.perspective else 0            
            min_z = view.z_shift * 0.1
            max_z = view.z_shift * 10
            ortho_height = 2. * view.z_shift * tan(view.fov * pi / 360)
        
            if fov != self.fov or min_z != self.min_z or max_z != self.max_z or ortho_height != self.ortho_height:
                self.fov = fov
                self.min_z = min_z
                self.max_z = max_z
                self.ortho_height = ortho_height
                self.resize_GL()
            
            
            if view.playing:
                view.adj('frame', view.frame_rate * elapsed)
                
                ncf = int(view.frame+1E-3) % len(view.frames)
                if ncf != cf:
                    changed = True
                    cf = ncf
                
                #CURRENT_FRAME = (CURRENT_FRAME + FRAME_RATE * elapsed) % len(FRAMES)
        
            if view.autorotate:
                view.adj('R', (0, view.autorotate_speed * elapsed, 0))
                changed = True
                #R = rot_y(, R)
    
    
        if changed or SCREENSHOT:
            
            
            switch_lighting(view.render_type, view.brightness)
        
            
            shaders.glUseProgram(0)
            if view.background_color <= 1:
                glClearColor(view.background_color, view.background_color, view.background_color, 0.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)      # Clear The Screen And The Depth Buffer
                line_color = ones(3) - view.background_color
            if view.background_color == 2:
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)      # Clear The Screen And The Depth Buffer
                self.fade_background((0.5, 0.5, 1.0, 1.0), (0.0, 0.0, 0.3, 1.0))
                line_color = (0, 0, 0)
                

 
                
            shaders.glUseProgram(self.render_shaders[int(view.render_type)])
    
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            
            R = view.R
            
            if view.multi_frame:
                draw_frames = [(f, None) for f in (cf + (1+view.multi_frame_skip) * arange(int(view.multi_frame))) % len(view.frames)]
            else:
                draw_frames = [(cf, None)]
                
            if view.tile > 0:
                shift = arange(view.tile+1) - 0.5*view.tile
                draw_frames = [(f, (x, y, z))
                    for (f, dummy) in draw_frames
                    for x in view.tile_x * shift
                    for y in view.tile_y * shift
                    for z in view.tile_z * shift
                ]

            #print draw_frames
            
            glColor4f(1.0, 1.0, 1.0, 1.0)

            if view.colorize:
                c0 = view.colorize - 1
                colors = [COLORS[(c0 + i) % (len(COLORS)-1) + 1] for i in range(len(draw_frames))]
            else:
                colors = [None] * len(draw_frames)
            
            if view.display_3d:
                position_scene(rot_y(view.eye_split, R))
                glColorMask(True, False, False, True)
                for (f, offset), color in zip(draw_frames, colors): draw_mesh(view.frames[f], offset, color=color)
    
                glClear(GL_DEPTH_BUFFER_BIT)
                position_scene(rot_y(-view.eye_split, R))
                glColorMask(False, True, True, True)
                for (f, offset), color in zip(draw_frames, colors): draw_mesh(view.frames[f], offset, color=color)
    
                glColorMask(True, True, True, True)
            
            else:
                position_scene(R)
                view_box = (view.XD,view.XU,view.YD,view.YU,view.ZD,view.ZU)
#                for (f, offset), color in zip(draw_frames, colors): draw_mesh(view.frames[f].crop(view_box), offset, color=color)
                for (f, offset), color in zip(draw_frames, colors): draw_mesh(view.frames[f], offset, color=color)
            glDisable(GL_LIGHTING)
    
            shaders.glUseProgram(0)
        
            glEnable( GL_LINE_SMOOTH )
            glEnable( GL_POLYGON_SMOOTH )
            glHint( GL_LINE_SMOOTH_HINT, GL_NICEST )
            glHint( GL_POLYGON_SMOOTH_HINT, GL_NICEST )
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            if view.draw_box:  #Draw box outline
                
                glLineWidth(VIEW.box_linewidth)
                glEnableClientState(GL_VERTEX_ARRAY)
                X0 = array([view.box_x0, view.box_y0, view.box_z0])
                X1 = array([view.box_x1, view.box_y1, view.box_z1])
                ub = (UNIT_BOX * (X1 - X0) + X0).astype('f')

                glColor4f(line_color[0], line_color[1], line_color[2], 1)
                glEnable(GL_DEPTH_TEST)
                glEnable(GL_BLEND)
                glVertexPointer(3, GL_FLOAT, 0, ub - 0.5 * (X0 + X1))
                
#                bz = (BOX_EDGE_CENTERS[..., newaxis] * view.R).sum(-1)
#                glDrawElements(GL_LINES, 2*len(BOX_EDGES), GL_UNSIGNED_INT, BOX_EDGES[argsort(bz[..., 2])])                

                glDepthMask(False)
                glDrawElements(GL_LINES, 2*len(BOX_EDGES), GL_UNSIGNED_INT, BOX_EDGES)                
                glDepthMask(True)
                glDisableClientState(GL_VERTEX_ARRAY)
                
            glDisable(GL_DEPTH_TEST)

            glColor4f(line_color[0], line_color[1], line_color[2], 1)

            glLineWidth(1)

            if MOUSE_CAPTURED:
                position_scene()
    
                glBegin(GL_LINE_LOOP)
                draw_gl_circle(r=.5)
                glEnd()
                glBegin(GL_LINES)
                glVertex3f(.5, 0, 0)
                glVertex3f(-.5, 0, 0)
                glVertex3f(0, .5, 0)
                glVertex3f(0, -.5, 0)
                glEnd()
                
            if view.display_frame:
                self.render_text(0.01, 0.01, '%s' % (VIEW.labels[cf]))
                
    
            glDisable(GL_LINE_SMOOTH)
            glDisable(GL_POLYGON_SMOOTH)
    
            glutSwapBuffers()
    
            if SCREENSHOT:
                self.screenshot(SCREENSHOT)    
                SCREENSHOT = None
            
        else:
            time.sleep(1E-2)
    
        view.changed = False
    
        


    def render_text(self, x, y, s, height=0.03, mono_spaced=True):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        aspect_ratio = float(self.width) / self.height
        glOrtho(0, aspect_ratio, 0, 1, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        scale = height / 150
        glScalef(scale, scale, scale)
            
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
        glEnable(GL_LINE_SMOOTH)
        glLineWidth(2.0)
        glDisable(GL_DEPTH_TEST)
        font = GLUT_STROKE_MONO_ROMAN if mono_spaced else GLUT_STROKE_ROMAN
        glTranslatef(x/scale, y/scale, 0);
        for c in s:
            glutStrokeCharacter(font, ord(c))
        glPopAttrib()
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
 
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        
    def fade_background(self, c_b, c_t):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        aspect_ratio = float(self.width) / self.height
        glOrtho(0, 1, 0, 1, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
            
        glBegin(GL_QUADS)
        glColor4f(*c_b); glVertex3f(0, 0, 0)
        glColor4f(*c_b); glVertex3f(1, 0, 0)
        glColor4f(*c_t); glVertex3f(1, 1, 0)
        glColor4f(*c_t); glVertex3f(0, 1, 0)
        glEnd()
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
 
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()        
        
    
#def inner_draw():
#    glutSolidSphere(0.25, 10, 10)

def draw_mesh(m, offset=None, color=None):
    if not hasattr(m, 'normals'):
        m.force_flat_normals()

    glEnableClientState(GL_VERTEX_ARRAY)
    if offset is not None: points = m.points + offset
    else: points = m.points

    glVertexPointer(3, GL_DOUBLE, 0, points)    
    
    if hasattr(m, 'normals'):
        glEnableClientState(GL_NORMAL_ARRAY)
        glNormalPointer(GL_DOUBLE, 0, m.normals)        
    
    if color is not None:
        glPushAttrib(GL_COLOR_BUFFER_BIT)
        glColor(*color)
    elif hasattr(m, 'colors'):
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(4, GL_DOUBLE, 0, m.colors)        
    
    glDrawElements(GL_TRIANGLES, 3 * len(m.triangles), GL_UNSIGNED_INT, m.triangles)
    
    glDisableClientState(GL_VERTEX_ARRAY)
    if hasattr(m, 'normals'): glDisableClientState(GL_NORMAL_ARRAY)
    if color is not None: glPopAttrib() #GL_COLOR_BUFFER_BIT)  
    elif hasattr(m, 'colors'): glDisableClientState(GL_COLOR_ARRAY)



def position_scene(r=eye(3)):
    global VIEW

    glLoadIdentity()                                        # Reset The View
    #if not VIEW.fov:
    #    glScalef(1.25/VIEW.z_shift, 1.25/VIEW.z_shift, 1.25/VIEW.z_shift)
    #    glTranslatef(0, 0, -10)
    #else:
    
    #Ortho scaling handled in view setup now, instead of by matrix
    glTranslatef(0.0, 0.0, -VIEW.z_shift)                      # Move Into The Screen
        
    M = eye(4, dtype='f')
    M[:3, :3] = r
    glMultMatrixf(M)
    




def find_new_file(name_str, max=10000):
    for i in range(max):
        if not os.path.exists(name_str % i):
            return name_str % i
    else:
        raise RuntimeError('Failed to find unused filename for "%s"' % name_str)


MOVIE_FILE = None


def format_array(a, precision=3):
    fmt = '%% 1.%sf' % precision
    if len(a.shape) > 1:
        return '(' + ', '.join(format_array(aa) for aa in a) + ')'
    else:
        return '(' + ','.join(fmt % aa for aa in a) + ')'
    

LAST_MOVIE_FRAME = 0


def key_pressed_exit_only(*args):
    global VIEW, FULLSCREEN, WINDOW

    k = args[0].lower()
    
    if k == ESCAPE:
        sys.exit()
        
        
def key_pressed(*args):
    global VIEW, FULLSCREEN, WINDOW, SCREENSHOT

    k = args[0].lower()
    
    if k == ESCAPE:
        sys.exit()       
            
    elif k == b'1':
        fn = find_new_file(os.path.join(VIEW.basedir, '%s_%%d.png' % VIEW.labels[int(VIEW.frame)]))
        print("Saving screenshot to {:s}".format(fn))
        SCREENSHOT = fn
        #WINDOW.screenshot(fn)
        
    elif args[0] == b'\t':
        FULLSCREEN = not FULLSCREEN
        WINDOW.fullscreen(FULLSCREEN)
        VIEW.changed = True

    elif args[0] == b'm':
        params = VIEW.param_dict(format=True)
        cf = int(VIEW.frame)
    
        if not hasattr(VIEW, 'movie_file'):
            fn = find_new_file(os.path.join(os.getcwd(), 'movie_%d.txt'))
            print("Created movie file: {:s}".format(fn))
            
            with open(fn, 'wt') as f:
                VIEW.output_frame_rate = VIEW.frame_rate
                
                f.write('''#Movie Description File
#Autogenerated on: %s
#-------- Format Description ---------
#
# Movie files MUST begin with "#Movie"
#
# Most lines are "[key]: [value]" pairs
#
# Lines that generate frames are "[play/pause]: [delta] [unit]",
#   where [delta] is the time change, and [unit] is either "frames" or "s":
#     play: 10 frames
#     pause: 1.0 s
#  (frames is in terms of frames in the input, not the output )
#
# Lines following play/pause that begin with "*" indicate they should be
#   smoothly varied in the video, e.g. to rotate the view:
#     pause: 1.0 s
#        * R: [(0, 1, 0), (0, 0, 1)]
#
# There is also the special parameter "* spin [number], (vx, vy, vz)", which
#   rotates the view [number] times about the vector (vx, vy, vz) during the
#   section.
#
# The parameter "output_frame_rate" specifies how many images to generate per
#   second of display time -- if output_frame_rate != frame_rate then it will
#   not generate one output frame per input data frame.
#
# Comments begin with "#".
#
#--------------- Setup ---------------
source: %s
output_frame_rate: %s


''' % (time.strftime("%Y %b %d %a %H:%M:%S %Z"), VIEW.sourcefiles, VIEW.output_frame_rate))
    
                VIEW.movie_file = fn
                
                for k, v in sorted(params.items()):
                    if k not in VIEW.hidden_params:
                        f.write('%s: %s\n' % (k, v))
                        
                f.write('\n#----------- Start of Movie -----------\n')
        else:
            static_lines = []
            dynamic_lines = []
            
            delta = cf - VIEW.last_frame
            
            print('Adding to {:s}: delta={:d}  '.format(VIEW.movie_file, delta),end ='')
            
            for k, v in params.items():
                if v == VIEW.last_params[k]: continue #This didn't change.
                if k in VIEW.hidden_params: continue
                if k == 'frame': continue #Handled seperately
                elif k in VIEW.smooth_params:
                    dynamic_lines.append('  * %s: %s\n' % (k, v))
                else:
                    static_lines.append('%s: %s\n' % (k, v))
                
                print('{:s} -> {:s}'.format((k, v),end = ''))
                    
            print('\n')
        
            with open(VIEW.movie_file, 'at') as f:
                if static_lines:
                    f.write(''.join(static_lines))
                    f.write('\n')
                    
                if delta:
                    f.write('play: %d frames\n' % delta)
                else:
                    f.write('pause: 1.0 s\n')
            
                if dynamic_lines:
                    f.write(''.join(dynamic_lines))
                
                f.write('\n')
            
    
        VIEW.last_params = params
        VIEW.last_frame = cf
        
    else: VIEW.key_press(k)
    
    
def mouse_func(button, state, x, y):
    global MOUSE_CAPTURED, MOUSE_X, MOUSE_Y
    
    if button == GLUT_LEFT:
        if state == GLUT_DOWN:
            MOUSE_CAPTURED = True
            MOUSE_X = x
            MOUSE_Y = y
        else:
            MOUSE_CAPTURED = False
            VIEW.changed = True

        
def draw_gl_circle(center=zeros(3), x=(1, 0, 0), y=(0, 1, 0), r=1., np = 100):
    theta = arange(np) * (2*pi) / np
    C = cos(theta)
    S = sin(theta)
    x = array(x)
    y = array(y)
    for c, s in zip(C, S):
        glVertex3fv(center + r * (x * c + y * s))
        
    
def draw_gl_line():
    pass
        
def motion_func(x, y):
    global MOUSE_CAPTURED, MOUSE_X, MOUSE_Y, WINDOW, VIEW
    
    #, R, FOV, Z_SHIFT #XR, YR, ZR

    if MOUSE_CAPTURED:
        dx = x - MOUSE_X
        dy = y - MOUSE_Y
        MOUSE_X = x
        MOUSE_Y = y
        
        #print dx, dy
        if VIEW.fov:
            w = WINDOW.height * tan(0.5 * VIEW.fov*pi/180) / VIEW.z_shift * 1.45 #WTF is the 1.45?  I don't really know...
        else:
            w = WINDOW.height / float(VIEW.z_shift) *0.5
        #print (x, y, w)
        #w = 300.
        dx /= w
        dy /= w
        x = (x - WINDOW.width / 2.) / w
        y = (y - WINDOW.height / 2.) / w

        #print x, y, dx, dy

        #Prevent blowups... probably unnecessary, but doens't hurt
        if abs(x) < 1E-3: x = 1E-3
        if abs(y) < 1E-3: y = 1E-3
        
        r = sqrt(x**2 + y**2)
        #print r
        phi = arctan2(y, x)

        r_hat = array([cos(phi), sin(phi)])
        phi_hat = array([-sin(phi), cos(phi)]) 
        
        dr = dot(r_hat, (dx, dy))
        dphi = dot(phi_hat, (dx, dy))

        if r > 1:
            dphi /= r
            r == 1.
        
        r_xy = r_hat * dr + (1 - r) * dphi * phi_hat
        r_z  = r * dphi

        VIEW.adj('R', (r_xy[1], r_xy[0], -r_z))


        
def special_pressed(*args):
    VIEW.key_press(args[0])
    

#Obsolete function, use "decode_ply" below.
#
#def unpack_ply(fn):
#    with open(fn, 'rt') as f:
#        lines = f.read().splitlines()
#    
#    def pl():
#        line = None
#        while not line: line = lines.pop(0).strip().lower()
#        else: return line
#    
#    if pl().startswith('ply') and pl().startswith('format ascii'):
#        pass
#    else: #Why not just use "not"s above?  Short circuiting of binary expressions!
#        raise ValueError("File is not ascii formatted PLY!  I don't know how to read it.")
#        
#    line = pl()
#    if not line.startswith('element vertex'):
#        raise ValueError("Expected 'element vertex [N]' to follow 'format ascii' in PLY")
#    
#    nv = int(line.split()[2])
#    
#    line = pl()
#    properties = []
#    while not line.startswith('element face'):
#        if line.startswith('end_header'):
#            raise ValueError("Didn't find 'element face [N]' line in PLY")
#        
#        if line.startswith('property'):
#            properties.append(line.split()[2])
#        
#        line = pl()
#        
#    nf = int(line.split()[2])
#    
#    line = pl()
#    if not line.startswith('property list') and len(line.split()) == 5 and not line.split()[4] == 'vertex_indices':
#        raise ValueError("Didn't find 'property list [type] [type] vertex_indices' line in PLY")
#
#    if not pl().startswith('end_header'):
#        raise ValueError("Didn't find 'end_header' line in PLY following vertex type")
#    
#    try: vertex_data = fromstring('\n'.join(lines[:nv]), sep=' ').reshape((nv, len(properties)))
#    except: raise ValueError('Vertex data has wrong number of elements.')
#    try: face_data = fromstring('\n'.join(lines[nv:nv+nf]), dtype='u4', sep=' ').reshape((nf, 4))
#    except: raise ValueError('Face data has wrong number of elements (this function only accepts triangles!).')
#    
#    X = zeros((nv, 3))\
#    
#    X[:, 0] = vertex_data[:, properties.index('x')]
#    X[:, 1] = vertex_data[:, properties.index('y')]
#    X[:, 2] = vertex_data[:, properties.index('z')]
#    
#    if 'nx' in properties:
#        N = zeros((nv, 3))
#        N[:, 0] = vertex_data[:, properties.index('nx')]
#        N[:, 1] = vertex_data[:, properties.index('ny')]
#        N[:, 2] = vertex_data[:, properties.index('nz')]
#    else:
#        N = None
#    
#    return X, N, face_data[:, 1:].copy()


class PLYError(Exception): pass
    
PLY_FORMAT = re.compile(r'format\s+(\w+)\s+([0-9.]+)')
PLY_FORMAT_TYPES = {'ascii':'',
                    'binary_big_endian':'>',
                    'binary_little_endian':'<' }

PLY_ELEMENT = re.compile(r'element\s+(\w+)\s+([0-9]+)')
PLY_ELEMENT_TYPES = ('vertex', 'face', 'edge')

PLY_PROPERTY_LIST = re.compile(r'property\s+list\s+(\w+)\s+(\w+)\s+(\w+)')

PLY_PROPERTY = re.compile(r'property\s+(\w+)\s+(\w+)')
PLY_PROPERTY_TYPES = {
    'char':'i1', 'uchar':'u1',
    'short':'i2', 'ushort':'u2',
    'int':'i4', 'uint':'u4',
    'float':'f', 'double':'d'
}

PY_TYPE = lambda x: float if x in ('f', 'd') else int


NP_PLY = {}
for k, v in PLY_PROPERTY_TYPES.items(): NP_PLY[dtype(v)] = k


def decode_ply(f, require_triangles=True):
    '''Decode a ply file, converting all saved attributes to dictionary entries.
    
    **Note: this function usually does not need to be used directly; for
    converting ply's to meshes, use :func:`open_mesh`.  Alternatively, this
    function can be used if you require access to fields beyond the basic mesh
    geometry, colors and point normals.**
    
    For more info on the PLY format, see:
        * http://www.mathworks.com/matlabcentral/fx_files/5459/1/content/ply.htm
        * http://paulbourke.net/dataformats/ply/
    
    Parameters
    ----------
    f : string or file
        A valid PLY file
    require_triangles : bool (default: True)
        If true, converts ``element_data["face"]["vertex_indices"]`` to an array
        with shape ``([number of triangles], 3)``.  Raises a PLYError if there
        are not triangular faces.
        **Note: if ``require_triangles=False``, ``element_data["face"]["vertex_indices"]``
        will be a numpy array of data type ``"object"`` which contains arrays
        of possibly variable length.**
        
    Returns
    -------
    element_data : dict
        A dictionary of all the properties in the original file.
        The main dictinoary should contain two sub-dictionaries with names
        ``"vertex"`` and ``"face"``.
        These dictionaries contain all of the named properties in the PLY file.
        Minimal entries in ``"vertex"`` are ``"x"``, ``"y"``, and ``"z"``.
        ``"face"`` should at least contain ``"vertex_indices"``.
    '''
    if type(f) is str:
        f = open(f, 'rb')
    
    line = f.readline().strip()
    
    if line != b'ply':
        raise PLYError('First line of file %s is not "ply", this file is not a PLY mesh!\n[%s]' % (repr(f), line))
    
    format_type = None
    format_char = ''
    format_version = None
    line = ''
    
    elements = []
    element_info = {}
    current_element = None
    
    while line != 'end_header':
        line = f.readline().decode('ascii')
        if not line.endswith('\n'):
            raise PLYError('reached end of file (%s) before end of header; invalid file.' % repr(f))
        
        line = line.strip()
        if not line or line.startswith('comment'): continue
        
        m = PLY_FORMAT.match(line)
        if m:
            if format_type is not None:
                raise PLYError('format type in file %s specified more than once\n[%s]' % (repr(f), line))
            format_type = m.group(1)
            if format_type not in PLY_FORMAT_TYPES:
                raise PLYError('format type (%s) in file %s is not known\n[%s]' % (format_type, repr(f), line))
            format_char = PLY_FORMAT_TYPES[format_type]

            format_version = m.group(2)
            continue
        
        m = PLY_ELEMENT.match(line)
        if m:
            t = m.group(1)
            if t not in PLY_ELEMENT_TYPES:
                raise PLYError('element type (%s) in file %s is not known\n[%s]' % (t, repr(f), line))
            elif t in elements:
                raise PLYError('element type (%s) appears multiple times in file %s\n[%s]' % (t, repr(f), line))
            
            elements.append(t)
            element_info[t] = [int(m.group(2))]
            current_element = element_info[t]
            continue
        
        m = PLY_PROPERTY_LIST.match(line)
        if m:
            if current_element is None:
                raise PLYError('property without element in file %s\n[%s]' % (repr(f), line))
            
            tn = m.group(1)
            tl = m.group(2)
            
            if tn not in PLY_PROPERTY_TYPES:
                raise PLYError('property type (%s) in file %s is not known\n[%s]' % (tn, repr(f), line))
            if tl not in PLY_PROPERTY_TYPES:
                raise PLYError('property type (%s) in file %s is not known\n[%s]' % (tl, repr(f), line))
        
            current_element.append((m.group(3), (format_char + PLY_PROPERTY_TYPES[tn], format_char + PLY_PROPERTY_TYPES[tl])))
            continue
        
        m = PLY_PROPERTY.match(line)
        if m:
            if current_element is None:
                raise PLYError('property without element in file %s\n[%s]' % (repr(f), line))
            
            t = m.group(1)
            
            if t not in PLY_PROPERTY_TYPES:
                raise PLYError('property type (%s) in file %s is not known\n[%s]' % (t, repr(f), line))
            
            current_element.append((m.group(2), format_char + PLY_PROPERTY_TYPES[t]))
            continue    
        
        if line.startswith('end_header'):
            break
        else:
            raise PLYerror('invalid line in header for %s\n[%s]')
            
    element_data = {}
    fast_triangles = False
          
    for e in elements:
        et = element_info[e]
        
        ne = et.pop(0)
        #print e, et
        
        fast_decode = True
        
        dtype_list = []
        py_types = []
        
        
        if e == 'face' and len(et) == 1 and et[0][0] == 'vertex_indices' and require_triangles:
            #This mesh only has vertex indices AND we have require_triangles = True
            #Lets load the data assuming it's triangles, and check later
            dtype_list = [('nv', et[0][1][0]), ('v', '3' + et[0][1][1])]
            fast_triangles = True
            
        else:
            for name, t in et:
                if type(t) is tuple: #This element is a list.
                    fast_decode = False
                    dtype_list.append((name, 'O'))
                    py_types.append((name, list(map(PY_TYPE, t))))
                else:
                    dtype_list.append((name, t))
                    py_types.append((name, PY_TYPE(t)))

        dt = dtype(dtype_list)

        if fast_decode:
            if format_type == 'ascii':
                s = ''.join(f.readline() for n in range(ne))
                dat = genfromtxt(StringIO(s), dtype=dt)
            else:
                dat = fromfile(f, dtype=dt, count=ne)
        
        else:
            
            def conv(t, x):
                try: return t(x)
                except: raise PLYError('expected %s, found "%s" (in file %s)' % (t, x, repr(f)))
            
            def get(t, count=1):
                t = dtype(t)
                n = t.itemsize*count
                s = f.read(n)
                if len(s) != n:
                    raise PLYEror('reached end of file %s before all data was read' % repr(f))
                return fromstring(s, t, count=count)
            
            dat = zeros(ne, dtype=dtype)
            
            for i in range(ne):
                if format_type == 'ascii':
                    parts = f.readline().split()
                    
                    for name, t in py_types:
                        if not parts:
                            raise PLYError('when decoding %s #%s, not enough items in line (in file %s)' % (e, i, repr(f)))
                        if type(t) in (list, tuple):
                            nl = conv(t[0], parts.pop(0))
                            if len(parts) >= nl:
                                dat[name][i] = [conv(t[1], x) for x in parts[:nl]]
                                parts = parts[nl:]
                            else:
                                raise PLYError('when decoding %s #%s, not enough items for list (in file %s)' % (e, i, repr(f)))
                        else:
                            dat[name][i] = conv(t, parts.pop(0))
                        
                else:
                    for name, t in et:
                        if type(t) is tuple:
                            nl = get(t[0])
                            dat[name][i] = get(t[1], nl)
                        else:
                            dat[name][i] = get(t)
                        
        
        if fast_triangles and e == 'face':
            if (dat['nv'] != 3).any():
                raise PLYError("require_triangles=True, but file contains non-triangular faces")
            element_data[e] = {'vertex_indices':dat['v']}
        else:
            element_data[e] = dict((name, dat[name]) for name, t in et)


    if require_triangles:
        if not fast_triangles:
            v = list(element_data["face"]["vertex_indices"])
    
            if not (array(list(map(len, v))) == 3).all():
                raise PLYError("require_triangles=True, but file contains non-triangular faces")
            
            element_data["face"]["vertex_indices"] = array(v)

    return element_data



def all_in(k, d):
    for kk in k:
        if kk not in d: return False
    else:
        return True
    
    

class Mesh(object):
    def __init__(self, points=zeros((0, 3)), triangles=zeros((0, 3)), normals=None, colors=None):
        if type(points) == str:
            data = decode_ply(points)
        
            try: self.points = array([data['vertex'][k] for k in ('x', 'y', 'z')]).T.astype('d')
            except: raise PLYError('ply file %s has missing or invalid vertex position data' % fn)
            
            try: self.triangles = data['face']['vertex_indices'].astype('u4')
            except: raise PLYError('ply file %s has missing or invalid triangle data' % fn)
            
            for pl in (['red', 'green', 'blue', 'alpha'], ['red', 'green', 'blue']):
                if all_in(pl, data['vertex']):
                    colors = array([data['vertex'][k] for k in pl]).T
                    break
            else:
                colors = None
                
            pl = ('nx', 'ny', 'nz')
            if all_in(pl, data['vertex']):
                self.normals = array([data['vertex'][k] for k in pl]).T
            
            #self.points, N, self.triangles = unpack_ply(points)
            #if N is not None: self.normals = N
        
        else:    
            self.points = array(points, dtype='d')
            self.triangles = array(triangles, dtype='u4')
            if normals is not None:
                self.normals = array(normals, dtype='d')
            elif len(self.points) == 0:
                self.normals = zeros((0, 3)) #Empty mesh gets empty normals, so we can add the them later.
                
        if colors is not None:
            if isinstance(colors, ndarray) and colors.dtype == dtype('u1'):
                colors = colors.astype('d') / 255.
            else:
                colors = asarray(colors, dtype='d')   
            
            if colors.ndim == 1:
                colors = ones(len(self.points), dtype='d')[:, newaxis] * colors
            
            if colors.shape[1] == 1:
                self.colors = colors * ones(4, dtype='d')
                self.colors[:, 3] = 1. #Not transparent
            elif colors.shape[1] == 3:
                self.colors = ones((len(self.points), 4), dtype='d')
                self.colors[:, :3] = colors
            elif colors.shape[1] == 4:
                self.colors = colors
            else:
                raise ValueError('last axis of colors should have 1, 3 or 4 elements (I, RGB, or RGBA)')
                  
            self.colors = ascontiguousarray(self.colors)
                    
#            else:
#                self.colors = random.rand(len(points), 4).astype('d')
#                self.colors[:, 3] = 1. #Not transparent
 
        #print hasattr(self, 'normals'), len(self.points)
        #if (not hasattr(self, 'normals')) and len(self.points):
            
        #    self.force_flat_normals()
        
        
    def force_flat_normals(self):    
        n = self.generate_normals()
        self.normals = tile(n.reshape(-1, 1, 3), (1, 3, 1)).reshape(-1, 3)
        self.points = self.points[self.triangles].reshape(-1, 3)
        if hasattr(self, 'colors'):
            self.colors = self.colors[self.triangles].reshape(-1, 3)
        self.triangles = arange(len(self.triangles)*3).reshape(-1, 3)
        
            
    def inverted(self):
        if hasattr(self, 'normals'): n = -self.normals
        else: n = None
        
        return Mesh(self.points.copy(), self.triangles[:, ::-1].copy(), n, colors=getattr(self, 'colors', None))
            
    def translate(self, offset):
        return Mesh(self.points + offset, self.triangles.copy())
        
    def draw_triangles(self, draw_func, with_z=False, close=True, *args, **kwargs):
        for t in self.triangles:
            if close:
                t = hstack((t, t[0:1]))
            if with_z:
                x, y, z = self.points[t, :3].T
                draw_func(x, y, z, *args, **kwargs)
            else:
                x, y = self.points[t, :2].T
                draw_func(x, y, *args, **kwargs)
                 
    def copy(self):
        return Mesh(self.points.copy(), self.triangles.copy())
                    
    def volume(self):
        px, py, pz = self.tps(0).T
        qx, qy, qz = self.tps(1).T
        rx, ry, rz = self.tps(2).T
        
        return (px*qy*rz + py*qz*rx + pz*qx*ry - px*qz*ry - py*qx*rz - pz*qy*rx).sum() / 6.
        
    def is_closed(self, tol=1E-12):
        x, y, z = self.points.T
        m2 = self.copy()
        m2.points += 2 * array((max(x) - min(x), max(y) - min(y), max(z) - min(z)))
        v1 = self.volume()
        v2 = m2.volume()
        return abs((v1 - v2) / v1) < tol
        
    def __add__(self, other):
        if hasattr(other, 'points') and hasattr(other, 'triangles'):
            m = Mesh(
                points = vstack((self.points, other.points)),
                triangles = vstack((self.triangles, other.triangles + len(self.points)))
            )
            
            
            for attr, default in (['normals', None], ['colors', ones(4)]):
                sa = getattr(self, attr, None)
                ga = getattr(other, attr, None)
                #print '!', attr, '!'
                
                if sa is not None or ga is not None: #If one is not none, make defaults for the other
                #if True:
                    if sa is None and default is not None:
                        sa = (ones((len(self.points), 1)) * default).astype('d')
                    if ga is None and default is not None:
                        ga = (ones((len(other.points), 1)) * default).astype('d')
                
                if sa is not None and ga is not None:
                    setattr(m, attr, vstack([sa, ga]))
            
            return m
            
        else: raise TypeError('Can only add a Mesh to another Mesh')
                    
    def tps(self, n):
        return self.points[self.triangles[:, n]]
                    
    def generate_normals(self, normalize=False):
        n = cross(self.tps(1) - self.tps(0), self.tps(2) - self.tps(0))
        if normalize:
            return norm(n)
        else:
            return n
        
    def force_z_normal(self, dir=1):                
        inverted = where(sign(self.normals()[:, 2]) != sign(dir))[0]
        self.triangles[inverted] = self.triangles[inverted, ::-1]

    def save(self, fn, ext=None):
        if ext is None:
            ext = os.path.splitext(fn)[-1][1:]
        
        ext = ext.lower()
        if ext == 'ply':
            self.save_ply(fn)
        elif ext == 'stl':
            self.save_stl(fn)
        else:
            raise ValueError('Extension should be "stl" or "ply".')
            
    def save_stl(self, fn, header=None):
        output = open(fn, 'wb')
        if header is None:
            header = '\x00\x00This is an STL file. (http://en.wikipedia.org/wiki/STL_(file_format))'

        e = '<'

        output.write(header + ' ' * (80 - len(header)))
        output.write(struct.pack(e + 'L', len(self.triangles)))
        
        for t, n in zip(self.triangles, self.normals(True)):
            output.write(struct.pack(e + 'fff', *n))
            for p in t:
                x = self.points[p]
                output.write(struct.pack(e + 'fff', *x))
            output.write(struct.pack(e + 'H', 0))
                
        output.close()
            
    def save_ply(self, fn):
        output = open(fn, 'wt')
        output.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
element face %d
property list uchar int vertex_indices
end_header
''' % (len(self.points), len(self.triangles)))
        
        for p in self.points:
            output.write('%10.5f %10.5f %10.5f\n' % tuple(p))
            
        for t in self.triangles:
            output.write('3 %5d %5d %5d\n' % tuple(t))

        output.close()
                
    def rot_x(self, angle):
        if hasattr(self, 'normals'): n = rot_x(self.normals, angle)
        else: n = None
        return Mesh(rot_x(self.points, angle), self.triangles, n)
        
    def rot_y(self, angle):
        if hasattr(self, 'normals'): n = rot_y(self.normals, angle)
        else: n = None
        return Mesh(rot_y(self.points, angle), self.triangles, n)
    
    def rot_z(self, angle):
        if hasattr(self, 'normals'): n = rot_z(self.normals, angle)
        else: n = None
        return Mesh(rot_z(self.points, angle), self.triangles, n)

    def crop(self,box):
        xd, xu, yd, yu, zd, zu = box
        points_d = dict()
        new_points = self.points.shape[0] * [[0.,0.,0.]]
        new_points_counter = 0
        new_triangles = self.triangles.shape[0] * [[0,0,0]]
        new_triangles_counter = 0
        if self.colors is not None:
            new_colors = self.colors.shape[0]* [[0.,0.,0.]]
        else:
            new_colors = None
        if self.normals is not None:
            new_normals = self.normals.shape[0] * [[0.,0.,0.]]
        else:
            new_normals = None
        for i,pt in enumerate(self.points):
            if (xd<=pt[0]<=xu and yd<=pt[1]<=yu and zd<=pt[2]<=zu):
                new_points[new_points_counter] = pt
                points_d[i] = new_points_counter
            if self.colors is not None:
                new_colors[new_points_counter] = self.colors[i]
            if self.normals is not None:
                new_colors[new_points_counter] = self.normals[i]
            new_points_counter+=1
        new_points = new_points[:new_points_counter]
        if self.colors is not None:
            new_colors = new_points[:new_points_counter]
        if self.normals is not None:
            new_normals = new_points[:new_points_counter]
        for tg in self.triangles:
            if all([tg[i] in points_d for i in range(3)]):
                new_triangles[new_triangles_counter] = [points_d[tg[0]],points_d[tg[1]],points_d[tg[2]]]
                new_triangles_counter += 1
        new_triangles = new_triangles[:new_triangles_counter]
        return Mesh(new_points,new_triangles,colors = new_colors,normals = new_normals)          


def vector_angle(v1, v2, t):
    v1 = norm(v1)
    v2 = norm(v2)
    
    s = dot(cross(v1, v2), t) 
    c = dot(v1, v2)
    
    return arctan2(s, c)
    
def trace_line(path, thickness=0.5, sides=15, colors=None, cmap=None, clim=None, closed=True):
    num_points = len(path)
    
    if len(path) == 2:
        T = norm(path[1] - path[0])
        T = [T, T]
        closed = False
    else:
        Ts = norm(plus(path) - path)
        T = norm(Ts + minus(Ts)) #Get average tangent on each point
    
        if not closed: #Correct end points
            T[0] = norm(Ts[0])
            T[-1] = norm(Ts[-2])

    
    N = [eye(3)[T[0].argmin()]]
    
    for t in T:
        Nn = N[-1] - dot1(N[-1], t) * t
        if mag(Nn) != 0:
            N.append(norm(Nn))
        else:
            N.append(N[-1])

    N = N[1:]
    
    if closed:
        np = norm(N[-1] - dot1(N[-1], T[0]) * T[0]) #Wrap around to measure angle

    
        angle = vector_angle(N[0], np, T[0])
        
        B = cross(T, N)

        theta = arange(num_points) * angle / num_points
        c = cos(theta).reshape((num_points, 1))
        s = sin(theta).reshape((num_points, 1))
    
        N, B = c * N - s * B, s * N + c * B
        
    else:
        B = cross(T, N)
        
    phi = 2 * pi * arange(sides) / sides
    phi.shape += (1,)
    x = sin(phi)
    y = cos(phi)

    thickness = asarray(thickness)
    if not thickness.shape: thickness = thickness * ones(len(path))
    
    normals = -vstack([n * x + b * y for n, b in zip(N, B)])
    points = vstack([p + 0.5 * tt * (n * x + b * y) for p, n, b, tt in zip(path, N, B, thickness)])
    
    
    if colors is not None:
        colors = array(colors, 'd')
        
        if cmap is not None:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
                    
            if clim is None: clim = (colors.min(), colors.max())
                
            colors = ScalarMappable(Normalize(*clim), cmap).to_rgba(colors)
        else:
            if colors.shape[-1] == 3: #RGB, convert to RGBA
                c = ones(colors.shape[:-1] + (4,), dtype='d')
                c[..., :3] = colors
                colors = c                 
    
        colors = (ones((num_points, sides, 1)) * colors.reshape(-1, 1, 4)).reshape(-1, 4)
    
    m = make_tube(points, sides, normals=normals, colors=colors, closed=closed)
    if m.volume() < 0: m = m.inverted()

    return m
    

def trace_ribbon(X, N, width=1.0, thickness=None, NR=5, colors=None, cmap=None, clim=None, closed=True):
    NP = len(X)
    
    if thickness is None: thickness = width * 0.2
    
    thickness = (ones(len(X)) * thickness).reshape(-1, 1, 1)
    width     = (ones(len(X)) * width).reshape(-1, 1, 1)
    
    Ts = norm(plus(X) - X)
    T = norm(Ts + minus(Ts)) #Get average tangent on each point
    
    if not closed: #Correct end points
        T[0] = norm(Ts[0])
        T[-1] = norm(Ts[-2])

    N = norm(N - dot1(T, N) * T)
    B = cross(T, N)
    
    X = X[:, newaxis, :]
    N = N[:, newaxis, :]
    B = B[:, newaxis, :]
    
    phi = linspace(-pi/2, pi/2, NR)

 
    outline_norm = array([cos(phi), sin(phi)]).T
    outline = 0.5 * ((width - thickness) * (1, 0) + thickness * outline_norm)
    outline = concatenate([outline, -outline], axis=1)
    outline_norm = vstack([outline_norm, -outline_norm])

    points = X + N * outline[..., 0:1] + B * outline[..., 1:2]
    normals = N * outline_norm[:, 0, newaxis] + B * outline_norm[:, 1, newaxis]
    
    #print len(X)
    #print points.shape
    #print colors.shape


    if colors is not None:
        colors = array(colors, 'd')
        
        if cmap is not None:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
                    
            if clim is None: clim = (colors.min(), colors.max())
                
            colors = ScalarMappable(Normalize(*clim), cmap).to_rgba(colors)
        else:
            if colors.shape[-1] == 3: #RGB, convert to RGBA
                c = ones(colors.shape[:-1] + (4,), dtype='d')
                c[..., :3] = colors
                colors = c                 
    
        colors = (ones(points.shape[:2]+(1,)) * colors.reshape(-1, 1, 4)).reshape(-1, 4)


    
    return make_tube(points.reshape(-1, 3), NR*2, normals=normals.reshape(-1, 3), colors=colors, closed=closed)


#def trace_split_ribbon(X, N, width=1.0, thickness=None, NR=5, colors=array([(1, 0, 0, 1), (0.5, 0.5, 0.5, 1)], 'd')):
def trace_split_ribbon(X, N, width=1.0, thickness=None, NR=5, colors=array([(1, 1, 0, 1), (0, 0, 1, 1)], 'd'), closed=True):
    if NR%2 == 0: NR += 1
    
    NP = len(X)



    width = (ones(len(X)) * asarray(width).flatten()).reshape(-1, 1, 1)

    if thickness is None: thickness = width * 0.25

    thickness = (ones(len(X)) * asarray(thickness).flatten()).reshape(-1, 1, 1)
    
    
    Ts = norm(plus(X) - X)
    T = norm(Ts + minus(Ts)) #Get average tangent on each point
    
    if not closed: #Correct end points
        T[0] = norm(Ts[0])
        T[-1] = norm(Ts[-2])
        
    N = norm(N - dot1(T, N) * T)
    B = cross(T, N)
    
    X = X[:, newaxis, :]
    N = N[:, newaxis, :]
    B = B[:, newaxis, :]
    
    phi = linspace(-pi/2, pi/2, NR)

 
    outline_norm = array([cos(phi), sin(phi)]).T
    outline = 0.5 * ((width - thickness) * (1, 0) + thickness * outline_norm)
    outline = concatenate([outline, -outline], axis=1)
    outline_norm = vstack([outline_norm, -outline_norm])
    
    
    
    NH = (NR-1) // 2
    m = []
    for i, shift in enumerate([-NH, -(NR+NH)]):
        o_s = roll(outline, shift, axis=1)[:, :NR+1]
        n_s = roll(outline_norm, shift, axis=0)[:NR+1]
        
        points = X + N * o_s[..., 0:1] + B * o_s[..., 1:2]
        normals = N * n_s[:, 0, newaxis] + B * n_s[:, 1, newaxis]
        m.append(make_shell(points.reshape(-1, 3), NR+1, normals=normals.reshape(-1, 3), colors=colors[i], closed=closed))
    
    return m[0] + m[1]
    
    #c = zeros((NP, NR*2, 4))
    #c[:, :NR] = colors[0]
    #c[:, NR:] = colors[1]
    #c = roll(c, NR//2, axis=1)
    
    #return make_shell(points.reshape(-1, 3), NR*2, normals=normals.reshape(-1, 3), colors=c.reshape(-1, 4))    
    
    
def make_shell(points, nc, normals=None, colors=None, closed=True):
    N = int(len(points) // nc)
    if nc * N != len(points):
        raise ValueError('Number of points in a shell must be a multiple of the number around the circumference!')
    
    tris = zeros((N, nc-1, 2, 3), dtype='i')
    
    i00 = arange(N).reshape(-1, 1) * nc + arange(nc-1).reshape(1, -1)
    i10 = roll(i00, -1, axis=0)
    i01 = i00 + 1
    i11 = i10 + 1
    
    tris[..., 0, 0] = i00
    tris[..., 0, 1] = i11
    tris[..., 0, 2] = i10
    tris[..., 1, 0] = i00
    tris[..., 1, 1] = i01
    tris[..., 1, 2] = i11
    
    tris = tris.reshape(-1, 3)
    
    if not closed:
        tris = tris[:-(nc-1)*2]
    
    return Mesh(points, tris, normals, colors)
    

def make_tube(points, nc, normals=None, colors=None, closed=True):
    N = int(len(points) // nc)
    if nc * N != len(points):
        raise ValueError('Number of points in a shell must be a multiple of the number around the circumference!')
    
    tris = zeros((N, nc, 2, 3), dtype='i')
    
    i00 = arange(N).reshape(-1, 1) * nc + arange(nc).reshape(1, -1)
    i10 = roll(i00, -1, axis=0)
    i01 = roll(i00, -1, axis=1)
    i11 = roll(i10, -1, axis=1)
    
    tris[..., 0, 0] = i00
    tris[..., 0, 1] = i11
    tris[..., 0, 2] = i10
    tris[..., 1, 0] = i00
    tris[..., 1, 1] = i01
    tris[..., 1, 2] = i11

    tris = tris.reshape(-1, 3)
    
    if not closed:
        tris = tris[:-nc*2]
    
    return Mesh(points, tris, normals, colors)
    
    
#def make_tube(points, nc, cap=False, normals=None, colors=None):
#    N = int(len(points) // nc)
#    if nc * N != len(points):
#        raise ValueError('Number of points in a tube must be a multiple of the number around the circumference!')
#    tris = []
#
#    
#    for i in range(N - (1 if cap else 0)):
#        i0 = i * nc
#        i1 = ((i + 1) % N) * nc
#    
#        for j0 in range(nc):
#            j1 = (j0 + 1) % nc
#            tris += [(i0 + j0, i0 + j1, i1 + j0), (i1 + j0, i0 + j1, i1 + j1)]    
#        
#    if cap:
#        tris += list(make_cap(points[:nc], dir=-1))
#        tris += list(make_cap(points[-nc:], offset = len(points) - nc))
#        
#    m = Mesh(points, tris, normals, colors)
#    return m


def json_to_traces(fn):
    with open(fn, 'r') as f:
        info = json.load(f)

    if  '__file_type__' in info:
        ft = info.pop('__file_type__')
    
        if ft == 'JSON 3D PATH':
            info['traces'] = [asarray(info.pop('path', zeros((0, 3))))]
            for k in ['normal', 'thickness']:
                if k in info:
                    info[k] = [info[k]]
            
        elif ft == 'JSON 3D TANGLE':
            closed = []
            traces = []

            for path in info.pop('paths', []):
                path.pop('info')
                traces.append(asarray(path.pop('path', zeros((0, 3)))))
                
                for k, v in path.items():
                    info[k] = info.get(k, []) + [v]
                    

            info['traces'] = traces

        else: 
            raise ValueError('file does not appear to be a valid tangle or path file\n(__file_type__=%s)' % ft)
    traces = info.pop('traces')
    traces = [asarray(t) for t in traces]

    if 'normal' in info and 'normals' not in info:
        info['normals'] = info['normal']

    return traces, info

    
def torus_knot(p=2, q=3, a=0.5, points=100):
    phi = arange(0, points) * 2 * p * pi / points
    th = q * phi / p
    
    r = 1 + a * cos(th)
    
    points = zeros((points, 3))
    points[..., 0] = r * cos(phi)
    points[..., 1] = r * sin(phi)
    points[..., 2] = a * sin(th)
    
    return points

def D(x):
    x = array(x)
    return 0.5 * (roll(x, -1, 0) - roll(x, 1, 0)) 
    
    
commonprefix = os.path.commonprefix
def commonsuffix(l):
    return os.path.commonprefix([x[::-1] for x in l])[::-1]
    

def names_to_sequence(fns):
    prefix = commonprefix(fns)
    suffix = commonsuffix(fns)
    np = len(prefix)
    ns = len(suffix)
    
    fn_sub = [fn[np:-ns] for fn in fns]
    try:
        fn_i = [int(fn) for fn in fn_sub]
    except:
        return fns #This list couldn't be mapped to sequence, so don't try
    else:
        min_sub = fn_sub[fn_i.index(min(fn_i))]
        max_sub = fn_sub[fn_i.index(max(fn_i))]
        return [prefix + '[%s-%s]' % (min_sub, max_sub) + suffix]
    
def make_movie(f, scale=None):
    def error(s, i, line):
        print("Movie file error: {:s}".format(s))
        print("{:d} >> {:s}".format(i+1, line))
        sys.exit()


    default_params = View().param_dict()

    commands = [[0, default_params, None]]

    output_frame_rate = commands[-1][1]['frame_rate']
    frame_rate = output_frame_rate
    source = None
    playing = False
    
    for i, line in enumerate(open(f, 'rt').read().splitlines()):
        ol = line
        if '#' in line:
            line = line[:line.find('#')]
            
        if not line: continue
            
        if ':' not in line: error('expected [key]:[value] pair', i, line)
            
        k, v = map(str.strip, line.split(':', 1))
        
        if k in ('play', 'pause'):
            try:
                delta, unit = map(str.strip, v.split())
                delta = float(delta)
            except: error("couldn't evaluate play/pause time step", i, line)
                        
            if unit == 's':
                num_frames = int(delta * output_frame_rate)
                delta_f = int(delta * frame_rate)
                
            elif unit == 'frames':
                num_frames = abs(int(delta * output_frame_rate / frame_rate))
                delta_f = int(delta)
            else: error("play/pause unit should be 's' or 'frames'", i, line)

            if k == 'pause': delta_f = 0
            
            commands.append([num_frames, {'frame':commands[-1][1]['frame']}, None])
            commands[-1][1]['frame'] += delta_f
            playing = True
            
        else:    
            try:
                v = eval(v)
            except: error("couldn't evaluate value: '%s'" % v, i, line)
            
            if k == 'output_frame_rate':
                output_frame_rate = v
            elif k == 'source':
                if source is not None: error('more than one source found!')
                source = sequence_to_names(v)
            else:
                if k.startswith("*"):
                    if not playing: error('found smoothly varying key not associated with play/pause')
                    k = k.strip('*').strip()
                else:
                    if playing:
                        commands.append([0, {'frame':commands[-1][1]['frame']}, None])
                        playing = False
            
                if k == 'spin':
                    if not playing:
                        error("spin only valid in play/pause section (use * spin)", i, line)
                    else:
                        num_spins, V = v
                        V = norm(V) * num_spins * 2 * pi
                        commands[-1][2] = V
                        
                elif k not in default_params:
                    error("unknown key '%s'" % k, i, line)
                else:
                    if k == 'R':
                        v = array(v)
                    commands[-1][1][k] = v
                    if k == 'frame_rate': frame_rate = v

    for i, d, sv in commands:
        print('Frames: {:d}'.format(i))
        if sv is not None:
            print('   spin -> {:s}'.format(sv))
        for k, v in d.items():
            print('   %s -> %s' % (k, v))
        
    current = commands[0][1]
    views = []
    
    
    for i, d, sv in commands:
        R0 = normalize_basis(current['R'])
        R1 = normalize_basis(d.get('R', R0))
        if rms(R0 - R1) < 1E-6:
            R_func = lambda x: R0
        else:
            Rn = array([normalize_basis(R0*(1-x) + R1*x) for x in linspace(0, 1, 21)])
            zn = zeros(len(Rn))
            dR = Rn[1:] - Rn[:-1]
            dR = cumsum(sqrt((dR**2).sum(-1).sum(-1)))
            zn[1:] = dR / dR[-1]
            R_func = interp1d(zn, Rn, axis=0)
        
        for n in range(i):
            x = float(n) / i
            if not len(views):
                views.append(current.copy())
                print(current['frame'])
            else:
                views.append({})
            for k, v1 in d.items():
                if k != 'R':
                    v0 = current[k]
                    views[-1][k] = v0 * (1.0-x) + v1 * x
                
            if sv is not None:
                views[-1]['R'] = rotate_by_vector(R_func(x), sv*x)
            else:
                views[-1]['R'] = R_func(x)                

        current.update(d)
        if sv is not None:
            current['R'] = rotate_by_vector(R1, sv)
        else:
            current['R'] = R1
            
            
    output_dir = os.path.splitext(f)[0]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    #for i, d in enumerate(views):
    #    print i, d['frame']
    show(source, scale=scale, movie_views=views, movie_image_name=os.path.join(output_dir, '%08d.tga'))
        


def sequence_to_names(seq):
    fns = []
    
    for fn in seq:
        if not os.path.exists(fn):
            start = fn.find('[')
            end = fn.find(']')
            if start >= 0 and end >= 0:
                sub = fn[start+1:end]
                suffix = fn[end+1:]
                prefix = fn[:start]
                try: start, end = sub.split('-')
                except:
                    fns.append(fn)
                else:
                    if len(start) == len(end):
                        repl = prefix + '%%0%dd' % len(start) + suffix
                    else:
                        repl = prefix + '%d' + suffix
                        
                    for i in range(int(start), int(end)+1):
                        fn = repl % i
                        if os.path.exists(fn):
                            fns.append(fn)
            else:
                fns.append(fn)
        else:
            fns.append(fn)
    
    return fns
    
if __name__ == '__main__':
    import pickle    
    import argparse
    
    m = []

    parser = argparse.ArgumentParser(description='Display 3D tangle files or meshes')
    parser.add_argument('files', metavar='files', type=str, nargs='+', help='Files to display; should be .json, .tangle, or .ply types.')
    parser.add_argument('-s', dest='scale', type=float, help='Scale factor for input [automatically determined]', default=None)
    args = parser.parse_args()    

    files = []
    for fn in args.files:
        if '*' in fn or '?' in fn:
            files += sorted(glob.glob(fn))
        else:
            files.append(fn)

    if open(files[0],'rb').read(6).lower() == b'#movie':
        if len(files) > 1:
            print("Warning: when making a movie, all files after the first are ignored...")
        make_movie(files[0], scale=args.scale)
    else:
        VIEW.basedir = os.path.split(files[0])[0]
        VIEW.sourcefiles = names_to_sequence(files)
        show(files, scale=args.scale)
