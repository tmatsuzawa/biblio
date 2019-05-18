import os
from flowtrace import flowtrace

# Frames to merge
framestomerge = 100

# Image Directory and Output Directory
image_dir = '/Volumes/labshared3-1/takumi/2018_07_22_wide/Tiff_folder/PIV_fv_vp_left_macro55mm_fps2000_Dp56p57mm_D12p8mm_piston10p5mm_freq3Hz_v300mms_setting1_File/'
out_dir = '/Volumes/labshared3-1/takumi/2018_07_22_wide/Tiff_folder/flowtrace_output_%dframestomerge_color' %framestomerge

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# standard
flowtrace(image_dir, framestomerge, out_dir, ext='tiff')
# subtruct median
#flowtrace(image_dir, framestomerge, out_dir, subtract_median=True,  ext='tiff')
# Color traces by time
#flowtrace(image_dir, framestomerge, out_dir, color_series=True,  max_cores=8, ext='tiff')
# parallel processing with 8 cores
# flowtrace(image_dir, framestomerge, out_dir, max_cores=2,  ext='tiff')