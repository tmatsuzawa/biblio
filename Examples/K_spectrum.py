

import stephane.mdata.M_manip as M_manip
import stephane.display.vfield as vfield


date = '2016_01_07'
dataDir = '/Volumes/labshared/Stephane_lab1/Vortices/2016_01_07/PIV_data/PIVlab_ratio2_W64pix_PIV_vortex_Rjoseph_fps2000_f100mm_tube_d12mm_v125'

M = M_manip.load(date,dataDir)
