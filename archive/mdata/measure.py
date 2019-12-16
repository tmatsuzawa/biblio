# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 19:00:47 2015

@author: stephane
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:42:10 2015

@author: stephane
"""

import numpy as np
import browse
import ids
import pickle_m
import os.path

class M:
    def __init__(self,S,dataDir,mindex,ref=0):
        #A measure is associated to :
        # a Sdata (parameters of a cine file)
        # a PIV param file
        # a dataDir with the npz measurements
        
        #list of parameters, should be a dictionnary ?
            #will depend on the considered object. for now, the type is moving grid
        #could be read from the title ?
        self.Sdata=S
        self.param=self.Sdata.param
        self.id=ids.Id(S,S.id.index,mindex)
        
        ######### Data file
        #Data are not store by default in Sdata, the data load must be called explicitly
        self.dataDir=dataDir
        self.filePIV=''  # associate the PIV file parameter ?
        
        self.U_ref,self.im_index=self.load_ref()
        #U_ref has attribute x,y, Ux, Uy

        self.nx,self.ny,self.nt=self.set_dimensions()
        
        self.ft,self.timescale,self.fx=self.set_scales()
#        self.x,self.y,self.t=set_scales()
        self.x,self.y=self.space_axis()
        self.t=self.time_axis()
        
        #for now, any measure is a PIV one : should evolute in time !
        #t is a 1D array
        #x and y are 2D arrays
        #Ux and Uy are 3D arrays    
        #self.im_index=np.zeros(0)
        #self.t=np.zeros(0)

        #self.x=np.zeros((0,0))
        #self.y=np.zeros((0,0))
        #self.Ux=np.zeros((0,0,0))
        #self.Uy=np.zeros((0,0,0))
        #U, d2Ux, d2Uy, smoothed data ?
    def get_fileList(self,display=False):
        fileList,n=browse.get_fileList(self.dataDir+'/','npz',display)
        if n==0:
            print(("No PIV files found ! at "+self.dataDir))
        return fileList,n
        
    def load_ref(self):
        fileList,nt=self.get_fileList(True)
        if nt>0:
            S=np.load(fileList[0])
            
            U_ref={'x':[],'y':[],'ux':[],'uy':[]}
            U_ref['ux']=S['ux']
            U_ref['uy']=S['uy']
            U_ref['x']=S['x']
            U_ref['y']=S['y']
        else:
            U_ref=None
        S.close()
        
        im_index=[browse.get_number(name,"/PIVData",".npz",False) for name in fileList]
        return U_ref,im_index
        
    def set_dimensions(self):
        fileList,nt=self.get_fileList()
        nx,ny=np.shape(self.U_ref['ux'])
            
        return nx,ny,nt
    
    def set_scales(self):
        #ft is in ms, time elapsed between two consecutive images
        ft=1000./self.param.fps
        #timescale is the time resolution of the PIV  measurements
        num=browse.get_number(self.dataDir,"_at","fps")        
        if num>0:
            timescale=num*ft/1000.
        else:
            timescale=1    
            
        fx=self.param.fx
        
        return ft,timescale,fx
        
    def shape(self):
        return self.nx,self.ny,self.nt
        #read nx and ny from the ref picture,
        #read nt from the number of associated files

    def time_axis(self): 
        #generation of a time axis
        #from the known value of fps and the t0 instant, generate an axis of time
        #t0 represent the image number at t=0
        im0=self.param.im0
        if im0 is not None:
            #print("Set origin of times")
            t0=(self.im_index[0]-im0)*self.ft
        else:
            print("No time reference found, origin of time arbitrary chosen at the beginning")
            t0=0.
        #time axis in ms
        #suppose that the frame are regularly spaced (!)
        #define it from the m_index list ??
        t=np.arange(t0,t0+self.nt*self.ft,self.ft)
        return t
    
    def space_axis(self):
        x=self.U_ref['x']
        y=self.U_ref['y']        
        
        if self.param.x0 is None:
            print("No space reference found, origin of space arbitrary chosen at the up left corner")
            x0=0.
            y0=0.
        else:
            x0=self.param.x0
            y0=self.param.y0
            
        x=(x-x0)*self.fx
        
        #Downward y axis (!):
        y=-(y-y0)*self.fx
        #if Sdata.param.sideview=='bv'
        #    ...
        if not (self.param.angle==0):
            print('rotation of the data needed !')
            
        return x,y
        
    def loadData(self):
        fileList,n=self.get_fileList()
        if n==0:
            return False
        #update = True means no change if the files are already present in the file
        #from a directory where all the .npz are stored, load_PIVData generate the x,y,Ux,Uy fields of Sdata
        #use the Dir name to find the fps ??
        #scale from the PIV settings : to be upgraded to a PIVsettings object
        self.scale = 2.8 * 10**-5
            
        #save parameters here !
        print("Loading data ...")
        self.Ux=np.zeros(self.shape())
        self.Uy=np.zeros(self.shape())
        
        #additionnal processing on the velocity measurements for now due to wrong PIVsettings (fps ans xscale)
        fact_x=self.fx/self.scale
        fact_t=1/self.ft
        fact=fact_x*fact_t        
        
        print(n)
        for name in fileList:
            count=fileList.index(name)
            S=np.load(name)
           # print(fileList.index(name))            
            if n>100:
                if count%(n//100)==0:
                    print((str(count*100//n) + " %"))
            
            self.Ux[:,:,fileList.index(name)]=fact*S['ux']
            self.Uy[:,:,fileList.index(name)]=fact*S['uy']
            
            S.close()
                    
        print("Data loaded")
        return True
        
    def eraseData(self):
        self.Ux=np.zeros((0,0,0))
        self.Uy=np.zeros((0,0,0))
        
            #Write the Sdata in the indicate filename
    def write(self,data=False):
        ############# Depreciated
        print("Writing Measure in pickle file : only the PIV parameters are written")
        if not data:
            self.eraseData()
        #write the Sdata object in a .txt file using the package pickle
        #The filename is directly determined from the Identification number
        self.fileDir = self.Sdata.dirCine + "M_" + self.id.date + "/"
        if not os.path.isdir(self.fileDir):
            os.makedirs(self.fileDir)

        #default name for the file containing the data
        self.filename=self.fileDir+"M_" + self.id.get_id() + ".txt"
        pickle_m.write(self,self.filename)
        
        print("Data written")
    #Read the Sdata object from the indicate filename
    #static method, generate a Sdata from the loaded file
    @staticmethod
    def read(filename,data=False):
        m=pickle_m.read(filename)   
        
        if data:
            m.load_measure()
        return m
        
        
    def get_index(self,x0,y0):
    #return the closest index compare to the given position of the grid
    #should depend on the orientation : if 90 angle, i and j roles are inverted

        if self.Sdata.param.angle==90:
            ys=x0
            xs=y0
            labelx='y'
            labely='x'
        else:
            xs=x0
            ys=y0
            labelx='x'
            labely='y'
        
        #bounds in mm
        scale=1000
        
        x=self.x
        y=self.y
        minx=np.min(x)*scale     
        maxx=np.max(x)*scale        
        
        miny=np.min(y)*scale        
        maxy=np.max(y)*scale        
    
        print('bounds:')
        print((labelx+' in ['+str(minx)+', '+str(maxx)+'] mm'))
        print((labely+' in ['+str(miny)+', '+str(maxy)+'] mm'))
        
        boolList=[x0>=minx,x0<=maxx,y0>=minx,y0<=maxx]
        if all(boolList):
            j=np.argmin(np.abs(x[0,:]-xs))
            i=np.argmin(np.abs(y[:,0]-ys))
        else:
            print('position out of bound')
            i=-1
            j=-1
        
        return i,j    
        
    @staticmethod
    def getloc(date,index,mindex):
        Dir=M.getDir(date)
        filename=Dir+"M_"+date+"_"+str(index)+"_"+str(mindex)+".txt"
        return filename
        
    @staticmethod
    def getDir(date):
        Dir="/Volumes/labshared3/Stephane/Experiments/"+date+"/M_"+date+"/"
        return Dir
        
    @staticmethod
    def getloc_M(m):
        Dir=M.getDir(m.id.date)
        filename=Dir+"M_"+m.id.date+"_"+str(m.id.index)+"_"+str(m.id.mindex)+".txt"
        return filename
    