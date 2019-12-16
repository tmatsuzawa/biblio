from . import cine
from pylab import *
from numpy import *
from scipy import *
import matplotlib.pyplot as plt
import pickle
import colorsys

Heights = arange(50,105,5)

fyle_speed = open('/Users/Martin/Documents/Knots/Table-Height-35-40/velocities_time_7.pickle')
speed_data = pickle.laod(fyle_speed)

fyle_radii = open('/Users/Martin/Documents/Knots/Table-Height-35-40/collected_radii.pickle')
radii_data = pickle.load(fyle_radii)

for h in Heights:
    index = '%(ind)d' % \
            {"ind": h}
    
    temp_R = radii_data[1][index]
    temp_RS = radii_data[2][index]
    temp_Rt = radii_data[0][index]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.errorbar(temp_Rt, temp_R, yerr = temp_RS, c='b', fmt= 'o')
    ax1.set_xlabel('Time * V/R')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('Vortex Radius (r/R)', color='b')
    ax1.ylim([0, 1.3])
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
        
    ax2 = ax1.twinx()
    
    temp_V = speed_data[2][index]
    temp_VS = speed_data[3][index]
    temp_Vt = speed_data[1][index]

    ax2.errorbar(temp_Vt, temp_v, yerr = temp_VS, c='r', fmt = 'o')
    ax2.set_ylabel('Vortex Speed (R/sec)', color='r')
    ax2.ylim([0, 10])
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    plt.show()