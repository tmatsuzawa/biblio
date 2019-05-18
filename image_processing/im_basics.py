from PIL import Image

def change_contrast(img, level, im_bit=8):
    """

    Parameters
    ----------
    img: PIL.Image class object
    level: contrast level, choose this to be 0-255 for 8-bit images

    Returns
    -------

    """
    bitsize = 2 ** im_bit
    int_max = bitsize - 1

    factor = (bitsize * (level + int_max)) / (int_max * (bitsize - level))

    def contrast(c):
        return bitsize / 2 + factor * (c - bitsize / 2)

    return img.point(contrast)