import matplotlib.pyplot as plt
import library.display.graph as graph
import glob
from PIL import Image
import numpy as np

tiff_dir = '/Volumes/labshared3-1/takumi/flash_test/Tiff_folder/flash_test_File/'
fps = 4000.0 #fps


tiffs = glob.glob(tiff_dir + '*.tiff')
tiffs = sorted(tiffs)

intensity = []
time = []
for i, tiff in enumerate(tiffs):
    print tiff
    im = Image.open(tiff)
    imarray = np.array(im)

    time.append(i/fps)
    intensity.append(imarray.sum())

fig1, ax1 = graph.scatter(time, intensity, fignum=1, subplot=121)
fig1, ax2 = graph.scatter(time, intensity, fignum=1, subplot=122)
ax2.set_lim(0, 0.1)
plt.show()