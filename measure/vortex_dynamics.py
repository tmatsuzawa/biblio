




import numpy as np
import stephane.vortex.track as track


import stephane.mdata.M_manip as M_manip
import stephane.tools.rw_data as rw_data
import glob
import stephane.tools.browse as browse
import stephane.pprocess.check_piv as check
import stephane.mdata.Mdata_PIVlab as pivlab
import stephane.mdata.Sdata_manip as Sdata_manip
import stephane.mdata.Sdata as Sdata
import stephane.analysis.cdata as cdata
import numpy as np
import stephane.pprocess.test_serie as tests
import stephane.display.graphes as graphes
import stephane.display.panel as panel
import stephane.vortex.track as track
import stephane.manager.access as access


def surdiffusion():
    
    #circulation (estimate)
    Gamma = 5 * 10**3 #mm^2/s^-2
    
    #radius (measured)
    d0 = 35. #mm

    #Urms of the background
    C = 5. * 10**3 / 15. #mm^2/s-^2
    C = 15**2    #mm^2/s^-2
    
    alpha = 2./3 *C*Gamma**2/d0**4
    
    print(str(alpha)+' s^3/mm^2')
    
def main():
    surdiffusion()
    
main()