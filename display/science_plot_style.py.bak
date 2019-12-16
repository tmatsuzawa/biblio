#!/usr/bin/env python
import pylab

'''Module for absolute control over axis instances, resolution, saving, etc of plots.
Author: Dustin Kleckner
with tweaks by Noah Mitchell
'''
# tight_layout()

# Matplotlib used to have a bug where this would need to be 72/90
font_ratio = 1.

font_size = 8

params = {
    'axes.labelsize': font_size  * font_ratio,
    'text.fontsize': font_size  * font_ratio,
    'legend.fontsize': font_size  * font_ratio,
    'xtick.labelsize': font_size  * font_ratio,
    'ytick.labelsize': font_size  * font_ratio,
    'axes.linewidth': 0.5,
    'legend.frameon':False,
    # 'text.usetex': True,
    'font.family': 'sans-serif',
    'font.size' : font_size  * font_ratio,
    # 'sans-serif': ['Helvetica'],
    # 'mathtext.fontset': 'stixsans',
     'xtick.major.size': 2,
     'xtick.minor.size': 1,
     'ytick.major.size': 2,
     'ytick.minor.size': 1,
     'lines.linewidth' : 0.5,
}

try:
    pylab.rcParams.update(params)
except AttributeError:
    import matplotlib
    matplotlib.rcParams.update(params)

SCALE_FACTOR = 1


def set_scale_factor(x):
    global SCALE_FACTOR
    SCALE_FACTOR = x

    p = {}
    for k, v in params.iteritems():
        if type(v) in (int, float):
            v = v * SCALE_FACTOR
        p[k] = v

    pylab.rcParams.update(p)


def figure_in_mm(width, height, **kwargs):
    figwidth = width/25.4 * SCALE_FACTOR
    figheight = height/25.4 * SCALE_FACTOR
    return pylab.figure(figsize=(figwidth, figheight), **kwargs)


def figure_in_pixels(width, height, dpi=150, **kwargs):
    figw = width * SCALE_FACTOR
    figh = height * SCALE_FACTOR
    return pylab.figure(figsize=(figw/dpi, figh/dpi), dpi=dpi, **kwargs)


def axes_in_mm(x0, y0, w, h, fig=None, label=None, label_xoff=1.5, label_yoff=1.5, label_params={}, **kwargs):
    """

    Parameters
    ----------
    x0
    y0
    w
    h
    fig
    label
    label_xoff
    label_yoff
    label_params
    ensure_even : bool
        round the size of the Figure such that it has even number of pixels, necessary for most versions of ffmpeg
    kwargs

    Returns
    -------

    """
    if fig is None:
        fig = pylab.gcf()
    size_mm = fig.get_size_inches() * 25.4
    fig_scale = SCALE_FACTOR/pylab.concatenate([size_mm, size_mm])

    ax = fig.add_axes([x0, y0, w, h] * fig_scale, **kwargs)
    if label is not None:
        lp = dict(va='top', ha='left', fontweight='bold')
        lp.update(label_params)
        ax.text(float(label_xoff)/w, 1-float(label_yoff)/h, label, transform=ax.transAxes, **lp)

    return ax


def joint_image_crop(*imgs):
    imgs = map(pylab.asarray, imgs)

    for img in imgs:
        if img.shape != imgs[0].shape:
            raise ValueError('joint_image_crop requires all images to have the same size')

    rows = pylab.array([(img == img[:, 0:1, :]).all(1).all(-1) for img in imgs]).all(0)
    cols = pylab.array([(img == img[0:1, :, :]).all(0).all(-1) for img in imgs]).all(0)

    ih, iw = imgs[0].shape[:2]

    y0 = 0
    y1 = ih
    while (rows[y0] and y0 < y1):
        y0 += 1
    while (rows[y1-1] and y1 > y0):
        y1 -= 1

    x0 = 0
    x1 = iw
    while (cols[x0] and x0 < x1):
        x0 += 1
    while (cols[x1-1] and x1 > x0):
        x1 -= 1

    return [img[y0:y1, x0:x1] for img in imgs]


def autocrop_image(img, border=0):
    img = pylab.asarray(img)

    ih, iw = img.shape[:2]

    rows = (img == img[:, 0:1, :]).all(1).all(-1)
    y0 = 0
    y1 = ih
    while (rows[y0] and y0 < y1): y0 += 1
    while (rows[y1-1] and y1 > y0): y1 -= 1

    cols = (img == img[0:1, :, :]).all(0).all(-1)
    x0 = 0
    x1 = iw
    while (cols[x0] and x0 < x1): x0 += 1
    while (cols[x1-1] and x1 > x0): x1 -= 1

    if border:
        x0 = max(x0 - border, 0)
        x1 = min(x1 + border, iw)
        y0 = max(y0 - border, 0)
        y1 = min(y1 + border, ih)

    return img[y0:y1, x0:x1]


def overlay_image(img, x0, y0, w=None, h=None, fig=None, expand=5,
                  interpolation='hamming', autocrop=True, va='bottom',
                  ha='left', **kwargs):
    """

    Parameters
    ----------
    img :
    x0 :
    y0 :
    w :
    h :
    fig :
    expand :
    interpolation :
    autocrop :
    va :
    ha :
    **kwargs : keyword arguments to pass to imshow
        Any additional options passed to imshow
    """
    img = pylab.array(img)

    if autocrop:
        img = autocrop_image(img)

    ih, iw = img.shape[:2]

    force_aspect = False

    if w and h:
        force_aspect = True
    elif w:
        h = float(w) * ih / iw
    elif h:
        w = float(h) * iw / ih
    else:
        raise ValueError('must specify width or height for overlay_image')

    if va == 'top':
        y0 -= h
    elif va == 'center':
        y0 -= 0.5*h
    elif va != 'bottom':
        raise ValueError('va should be one of "bottom", "top", or "center"')

    if ha == 'right':
        x0 -= w
    elif ha == 'center':
        x0 -= 0.5*w
    elif ha != 'left':
        raise ValueError('ha should be one of "left", "right", or "center"')

    if expand:
        ex = float(expand) / iw
        ey = float(expand) / ih
        x0 -= w * ex
        w *= 1 + ex
        y0 -= h * ey
        h *= 1 + ex

    ax = axes_in_mm(x0, y0, w, h, fig=fig)
    ax.imshow(img, interpolation=interpolation, clip_on=False, **kwargs)
    ax.set_xlim(-expand, iw+expand)
    ax.set_ylim(ih+expand, -expand)
    ax.set_axis_off()
    if force_aspect:
        ax.set_aspect('auto')

    return ax


def inset_image(img, x0, y0, w=None, h=None, ax=None, interpolation='hamming',
                expand=5, autocrop=True, va='bottom', ha='left', **kwargs):
    img = pylab.array(img)

    if autocrop:
        img = autocrop_image(img)

    ih, iw = img.shape[:2]

    if ax is None: ax = pylab.gca()
    fig = ax.get_figure()

    wf, hf = fig.get_size_inches()  # Real size of figure
    aspect = wf / hf  # Real aspect ratio

    xl = ax.get_xlim(); xr = float(xl[1] - xl[0])
    yl = ax.get_ylim(); yr = float(yl[1] - yl[0])

    ap = ax.get_position()
    awf = ap.x1 - ap.x0  # Axes height in figure units (0-1)
    ahf = ap.y1 - ap.y0

    x0a = (x0 - xl[0]) / xr  # x offset of inset in axes units
    y0a = (y0 - yl[0]) / yr
    x0f = x0a * awf + ap.x0  # X offset of inset in figure units
    y0f = y0a * ahf + ap.y0

    if w:
        w = w / xr * awf
    if h:
        h = h / yr * ahf

    force_aspect = False
    if w and h: force_aspect = True
    elif w: h = float(w) * ih / iw * aspect
    elif h: w = float(h) * iw / ih / aspect
    else: raise ValueError('must specify width or height for inset_image')

#    print w, h, x0, y0, x0f, y0f

    if va == 'top':
        y0f -= h
    elif va == 'center':
        y0f -= 0.5*h
    elif va != 'bottom':
        raise ValueError('va should be one of "bottom", "top", or "center"')

    if ha == 'right':
        x0f -= w
    elif ha == 'center':
        x0f -= 0.5*w
    elif ha != 'left':
        raise ValueError('ha should be one of "left", "right", or "center"')

    if expand:
        ex = float(expand) / iw
        ey = float(expand) / ih
        x0f -= w * ex
        w *= 1 + 2*ex
        y0f -= h * ey
        h *= 1 + 2*ey

    ax = fig.add_axes([x0f, y0f, w, h])

    ax.imshow(img, interpolation=interpolation, clip_on=False, **kwargs)
    ax.set_xlim(-expand, iw+expand)
    ax.set_ylim(ih+expand, -expand)
    ax.set_axis_off()
    if force_aspect:
        ax.set_aspect('auto')

    return ax


if __name__ == '__main__':
    fig = figure_in_mm(200, 100)

    x = pylab.linspace(0, 20, 100)
    y = pylab.sin(x)

    axes = [
            axes_in_mm(x0, y0, width, height, label=part)
            for x0, y0, width, height, part in (
                [20, 20, 100, 60, 'A'],
                [130, 55, 50, 25, 'B'],
                [130, 20, 50, 25, 'C'],
            )
        ]

    for ax in axes:
        ax.plot(x, y)

    fig.savefig('plot_example.pdf')