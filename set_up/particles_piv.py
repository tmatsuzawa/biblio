# -*- coding: utf-8 -*-



import numpy as np


def volume():
    
    ### parameters :
    V = 0.3*10**(-3) #Volume of the total box in m^3
    D = 10*10**(-6)  #particle diameter in m
    
    dx = 10*10**(-6)  #pixel size in microns
    
    
    #surface of a piv box
    W = 32 # width of the PIV box
    S = (W*dx)**2 #surface in m**2
    dh = 0.2*10**(-3) # width of the laser sheet. default is 200µm (a little bit arbitrary)
    Vbox = S*dh  #volume of a box
    
#    number of boxes 
    N = V/Vbox
    Npart = 10*N
    
    #mass of particles :
    rho = 10**3
    Vpart = np.pi/6.*D**3
    m = Vpart*Npart*rho*10**3 #in g
    
    print(Npart)
    print("mass of particles : "+str(round(m*10000)/10000)+'g')
    
    
def main():
    volume()
    
main()