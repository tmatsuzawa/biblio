import cine
from pylab import *
from numpy import *
import scipy
import scipy.interpolate
import matplotlib.pyplot as plt
import pickle


file = '/Users/Martin/Documents/Knots/Table-Height-35-40/2012_08_09_rf_b6_b35_%(heights)dcm_%(run)d_Run00%(run)d-pc.s4d' % \
    {"heights": 95, "run": 1}    
#file = '/Users/Martin/Documents/Knots/Table-Height-35-40/2012_08_09_rf_b6_b35_%(heights)dcm_%(run)d_Run00%(run)d-pc.s4d' % \
#    {"heights": Heights[j], "run": r}

time_scale = 180.5 #frames/sec
space_scale = 0.4 #mm/pixel
a = []
b = []
frame_seed = 50
frame_cutoffs = 55
AOA = 35
  
data = cine.open(file)
frame_end  = frame_cutoffs
frame_start = frame_seed
frame_step = 1

frame_range = arange(frame_start,frame_end,frame_step)
winner = []
center = []

for n in frame_range:
        vote = []
        location = []
        a_volume = data[n]
        nz,ny,nx = a_volume.shape
        a_volume = a_volume**2
        projection = sum(a_volume,1)
        max_int = projection.max()
        xy = where(projection>max_int*0.2)
        
        scatter(xy[0],xy[1],'o')
        show()
        
        theta = map(lambda x: atan(x[1]/x[0]), xy)
        
        plot(theta)
        show()
        
        