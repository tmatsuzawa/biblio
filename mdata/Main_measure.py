# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:43:58 2015

@author: stephane
"""

import stephane.mdata.M_manip as M_manip
import stephane.mdata.Sdata_measure as Sdata_measure
import pylab as plt
import numpy as np
import stephane.display.graphes as graphes


def main_measure(M):
    plt.close('all')
    
    Dt=50
    N=100

    #multi_plane_measurements(Dt,N)
    vertical_spreading(M,Dt,N)

def vertical_spreading(M,Dt,N):
    n=100
    ti=1
    
    t=np.zeros(n)
    z_i=np.zeros(n)
     
    for i in range(n):
        print(i)
        t0=(i+1)*N+ti
        mid=int(t0+Dt/2)
        print(mid)
        t[i]=M.t[mid]   
        
        z,Ux_moy,Uy_moy=Sdata_measure.profile_average(M,t0,t0+Dt,1)
        z,Ux_rms,Uy_rms=Sdata_measure.profile_average(M,t0,t0+Dt,2)

        Ux_norm=Ux_rms*t[i]/100
        Uy_norm=Uy_rms*t[i]/100
        
        print(Ux_moy)
        print(Uy_moy)
        a=-2.4
        b=0.12        
  #      U_spatial=np.concatenate((a*z[z<0]+b,0*z[z>0]+b))
        
#        fig=i
#        plt.figure(fig)
                
        #find the z position where Ux_rms ~ 0.2 m/s
        # plot this z position as a function of time
        thres=0.05 
        indices=np.where(Ux_norm<thres)
        if len(indices)>0:
            indice=indices[0][0]
        else:
            indice=np.argmin(np.abs(Ux_norm-thres)) #Uy correspond to the Ux component for angle=90        
        z_i[i]=z[indice]
        
        graphes.graph(z,Ux_rms,fignum=1,label='g+--')
        graphes.graph(z,Uy_rms,fignum=-2,label='b^--')

   #     graphes.graph(z,Ux_moy,-1,'+--')
   #     graphes.graph(z,Uy_moy,0,'^--')
        graphes.graph([z_i[i],z_i[i]],[0,0.7],fignum=-2,label='r-')
   
        Dir=M.fileDir+'Velocity_Distribution'+'/'
        file=graphes.set_title(M,'t='+str(int(mid))+' ms'+'Urms_zprofile')
        filename=Dir+file
        
        print(Dir)
        print(filename)
      #  graphes.save_fig(fig,filename,Dir,'png') 
         

 #       graphes.graph(z-z_i[i],Ux_norm*1.8,0,'+--')
 #       graphes.graph(z-z_i[i],Uy_norm*1.8,0,'^--')
        
        # graphes.graph([z_i[i],z_i[i]],[0,1],0,'r-')
#        graphes.set_axes()
        graphes.legende('$z$ (m)','$U_{rms} (t/t_0)$','')
        graphes.set_title(M,'U_rms_vs_z')        
        
    #average on horizontal line (if rotated, correspond to vertical line)
#compute the vertical RMS velocity profile
    graphes.graph(t,z_i,-1,'ro')
    graphes.legende('$t$ (ms)','$z$ (m)','')

    graphes.graphloglog(t,z_i,-1,'ro')
    graphes.legende('$t$ (ms)','$z$ (m)','')


def multi_plane_measurements(Dt,N):
    ti=500

    n=13    
    #    Dtlist=[2,4,6,8,10,20,30,50,70,100,150,200,500]
    m=len(Mlist)
    
    t=np.zeros((n,m))
    Ux_rms=np.zeros((n,m))
    Uy_rms=np.zeros((n,m))
    Uxt_rms=np.zeros((n,m))
    Uyt_rms=np.zeros((n,m))

    Zlist=[M.param.Zplane for M in Mlist]
    
    Dtlist=[Dt]    
    
    for M in Mlist:  
        j=Mlist.index(M)
        print(j)
        for i in range(n):
            t0=(i+1)*N+ti
            mid=t0+Dt/2
            t[i]=M.t[mid]
    #        Sdata_measure.velocity_distribution(M_1000fps,t0,t0+Dt)
            if i==0:
                fig=j*2+1
                plt.figure(fig)
#            Ux_rms[i,j],Uy_rms[i,j],Uxt_rms[i,j],Uyt_rms[i,j]=Sdata_measure.velocity_distribution(M,t0,t0+Dt)
            
        Dir=M.fileDir+'Velocity_Distribution'+'/'
        file=graphes.set_title(M,'Dt='+str(Dt/10)+' ms'+'pdf_U')
        filename=Dir+file
        
        print(Dir)
        print(filename)
       # graphes.save_fig(fig,filename,Dir) 
         
        U_rms=(Ux_rms+Uy_rms)/2
        Ut_rms=(Uxt_rms+Uyt_rms)/2
        
    graphes.graph(Zlist,U_rms[0,:],-1,'o')
    graphes.graph(Zlist,Ut_rms[0,:],0,'^')
    
    graphes.graph(Zlist,U_rms[0,:],-1,'+--')
    for i in range(1,n):
        graphes.graph(Zlist,(U_rms[i,:]*t[i]/t[0]),0,'+--')
    
    graphes.set_axes(-110,50,0,1)
        
    graphes.legende('$Z (mm)$','$(t/t_0) <Urms_{xi}>_{Dt,x,y}$','')
    file=graphes.set_title(M,'U_rms_vs_t')
    filename=Dir+file

       # graphes.save_fig(fig+1,filename,Dir)        
  #  for M in Mlist:
  #      Sdata_measure.velocity_distribution(M,t0,t0+Dt)

def main_generation():
    global Mlist_v
    Mlist_v=[]
   # date='2015_03_21'  
      #  indexList=[0,2,4,5,17,6,16,7,12,14]
    date='2015_03_04'
    #M_manip.Measure_gen_day(date)

    indexList=[2,4,6,10]
    
    for index in indexList:
        print(indexList.index(index))
        Mlist_v.append(M_manip.load_Mdata(date,index,0))

#Mlist=[M0,M4,M5,M2]
#main_generation()
def main():
    for M in Mlist_2015_09_17:
        main_measure(M)

if __name__ == '__main__':
    main()