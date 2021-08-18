import numpy as np
import gsw as gsw
import xarray as xr
from matplotlib import pyplot as plt
from .threshold import threshold as _threshold
from .gradient import gradient as _gradient
from .holtetalley import holtetalley as _holtetalley
from .energy import mld_pe_anomaly as _mld_pe_anomaly_Bisection
from .energy_Newton import mld_pe_anomaly as _mld_pe_anomaly_Newton
from .energy import mld_delta_pe as _mld_delta_pe
from .pe_anomaly import pe_anomaly as _pe_anomaly

class column():
    
    def __init__(self,
                 kind='idealized',
                 idealized_type='linear',T0=0,dTdz=0,S0=0,dSdz=0,Sb0=0,Tb0=0,Smin=0,Tmin=0,mixedfrac=0.5,
                 Boussinesq=False,Compressible=False,EOS='Full',rho0=1025,grav=9.81,
                 nz=1,Dpt=1,
                 ArgoPath='/net3/bgr/Datasets/Argo/202011-ArgoData/dac/aoml/',ArgoID=3900660,NP=0,MaxP=1.e8,
                 zc=[],dZ=[],T=[],S=[],
                 Debug=False
                ):
        
        self.rho0=rho0
        self.grav=grav
        
        self.Bouss        = Boussinesq
        self.Compressible = Compressible
        self.EOS=EOS
        
        if kind=='input':
            if (np.isempty(zc) or 
                np.isempty(dZ) or 
                np.isempty(T) or 
                np.isempty(S)):
                print("Missing inputs for kind='input'")
                return
            self.zc = zc
            self.dZ = dZ
            self.T  = T
            self.S  = S
            
        if kind=='idealized':
            
            #Set from input
            self.nz   = nz
            self.Dpt  = Dpt
            self.T0   = T0
            self.S0   = S0
            self.dTdz = dTdz
            self.dSdz = dSdz
            self.Tmin = Tmin
            self.Smin = Smin
            self.Tb0  = Tb0
            self.Sb0  = Sb0
            self.idealized_type = idealized_type
            self.mixedfrac = mixedfrac
            #Build
            self.Idealized()
        
        elif kind=='Argo':
            self.valid = self.ReadArgo(ArgoPath,str(ArgoID),NP,MaxP)
            if self.valid:
                self.GridArgo()
            else:
                if Debug: print('Error reading from Argo')
                return
        
        
    def ReadArgo(self,ArgoPath,ArgoID,NProf,MaxP):
        """Set up the T/S distributions for a given T0, dTdz, and Tmin (similar for S)."""
        try:
            hndl = xr.open_dataset(ArgoPath+'/'+ArgoID+'/'+ArgoID+'_prof.nc').sel(N_PROF=NProf)
            DayQC = float(hndl.JULD_QC)
            PosQC = float(hndl.POSITION_QC)
            if (np.max(DayQC==np.array([1,2,5,8]))
                and 
                np.max(PosQC==np.array([1,2,5,8]))
               ):
                #QC
                SALTQC = np.array(hndl.PSAL_QC[:],dtype=float)
                TEMPQC = np.array(hndl.TEMP_QC[:],dtype=float)
                PRESQC = np.array(hndl.PRES_QC[:],dtype=float)

                #Computes in-situ density from T&S, T, or S
                LI = (((SALTQC==1)|(SALTQC==2)|(SALTQC==5)|(SALTQC==8))
                      &
                      ((TEMPQC==1)|(TEMPQC==2)|(TEMPQC==5)|(TEMPQC==8))
                      &
                      ((PRESQC==1)|(PRESQC==2)|(PRESQC==5)|(PRESQC==8)))

                if (np.sum(LI)>20):
                    self.p_argo = np.array(hndl.PRES[:][LI],dtype=float)*1.e4
                    self.S_argo = np.array(hndl.PSAL[:].values[LI],dtype=float)
                    self.T_argo = np.array(hndl.TEMP[:].values[LI],dtype=float)
                    
                    FL = self.p_argo<MaxP
                    if np.sum(FL)>10:
                        self.p_argo=self.p_argo[FL]
                        self.S_argo=self.S_argo[FL]
                        self.T_argo=self.T_argo[FL]
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False
        
        except:
            return False
        
        
    
    def GridArgo(self):
        self.pi = np.array([0.,]+list(self.p_argo))
        Ti = np.array([self.T_argo[0],]+list(self.T_argo))
        Si = np.array([self.S_argo[0],]+list(self.S_argo))
        self.pc = 0.5*(self.pi[1:]+self.pi[:-1])
        self.T = 0.5*(Ti[1:]+Ti[:-1])
        self.S = 0.5*(Si[1:]+Si[:-1])
        self.dp = self.pi[:-1]-self.pi[1:]
        self.GetRho()
        self.GetZ()
        
    def GetZ(self):
        self.dz = -self.dp/(self.grav*self.rho)
        self.zi = np.array([0.,]+list(-np.cumsum(self.dz)))
        self.zc = 0.5*(self.zi[1:]+self.zi[:-1])
        #print(-np.sum(self.dp),np.sum(self.dz),np.sum(self.rho*self.dz*self.grav))
    
    def Idealized(self):
        self.SetGrid()
        self.SetState()
        self.GetPressure()
        self.GetRho()

    
    def SetGrid(self):
        """Set up an initial grid with nz levels to depth H (H should be positive)"""
        self.zi=np.linspace(0,-self.Dpt,self.nz+1)
        self.zc=0.5*(self.zi[1:]+self.zi[:-1])
        self.dz=(self.zi[:-1]-self.zi[1:])

    def SetState(self):
        """Set up the T/S distributions for a given T0, dTdz, and Tmin (similar for S)."""
        if self.idealized_type =='linear':
            self.T=np.maximum(self.Tmin,self.T0+self.zc*self.dTdz)
            self.S=np.maximum(self.Smin,self.S0+self.zc*self.dSdz)
        elif self.idealized_type=='two-layer':
            mixed_z = -self.mixedfrac*self.Dpt
            self.T = (self.T0*(self.zc>mixed_z) + 
                      np.maximum(self.Tmin,self.Tb0+(self.zc-mixed_z)*self.dTdz)*(self.zc<=mixed_z)
                     )
            self.S = (self.S0*(self.zc>mixed_z) + 
                      np.maximum(self.Smin,self.Sb0+(self.zc-mixed_z)*self.dSdz)*(self.zc<=mixed_z)
                     )
        else:
            print('Error setting state, unrecognized option for idealized_type')
            return
        self.GetPressure()
        self.GetRho()

    def GetPressure(self):
        if self.Bouss:
            self.pi=-self.zi*self.grav*self.rho0
            self.pc=0.5*(self.pi[1:]+self.pi[:-1])
            self.dp = self.pi[:-1]-self.pi[1:]
        else:
            self.pi=-self.zi*self.grav*self.rho0
            self.pc=0.5*(self.pi[1:]+self.pi[:-1])
            for ii in range(100):
                self.GetRho()
                self.pi=np.array([0.,]+list(np.cumsum(self.dz*self.grav*self.rho)))
                self.pc=0.5*(self.pi[1:]+self.pi[:-1])
            self.dp = self.pi[:-1]-self.pi[1:]
    
    def GetRho(self):
        if self.EOS=='Full':
            self.rho=gsw.density.rho(self.S,self.T,self.pc/10000.)
            self.prho=gsw.density.rho(self.S,self.T,0.)
        if self.EOS=='Linear':
            self.rho=self.rho0+(self.S-35)*0.8-(self.T-10)*0.2
            self.prho=self.rho0+(self.S-35)*0.8-(self.T-10)*0.2
            
    def plot_state(self,MLDs=[]):
        fi,ax=plt.subplots(2,2,figsize=(5,6))
        
        ax.ravel()[0].plot(self.T,self.zc,'k-')
        ax.ravel()[1].plot(self.S,self.zc,'k-')
        ax.ravel()[2].plot(self.rho,self.zc,'k-')
        ax.ravel()[3].plot(self.prho,self.zc,'k-')

        ax.ravel()[0].set(ylabel='z',xlabel='T')
        ax.ravel()[1].set(ylabel='z',xlabel='S')
        ax.ravel()[2].set(ylabel='z',xlabel=r'$\rho$')
        ax.ravel()[3].set(ylabel='z',xlabel=r'$\rho_\theta$')
        
        fi.tight_layout()
        return fi,ax
        
    def threshold(self,coord='depth',var='prho',delta=0.03,ref=10):
        return _threshold.threshold_mld_fixedcoord(-self.zc, self.prho, delta, ref)

    def gradient(self,coord='depth',var='prho',critical_gradient=1.0e-5):
        return _gradient.gradient_mld_fixedcoord(-self.zc, -self.prho, critical_gradient)
    
    def linearfit(self,coord='depth',var='prho',error_tolerance=1.0e-10):
        return _gradient.linearfit_mld_fixedcoord(-self.zc, -self.prho, error_tolerance)
    
    def holtetalley(self):
        return _holtetalley.algorithm_mld(self.pc/1.e4, self.S, self.T, self.prho)
    
    def mld_pe_anomaly(self,gradient=False,energy=10.0,iteration='Bisection'):
        if gradient:
            print('not ready for gradient in column mode')
            return -999
        else:
            dprhodz = self.prho*0.0
        if iteration=='Bisection':
            return _mld_pe_anomaly_Bisection(self.zc, self.dz, self.prho, dprhodz, energy).mld
        elif iteration=='Newton':
            return _mld_pe_anomaly_Newton(self.zc, self.dz, self.prho, dprhodz, energy).mld

    def mld_delta_pe(self,energy=10.0,Debug=False,eqstate='gsw'):
        return _mld_delta_pe(self.pc, self.dp,self.T,self.S, energy,Debug=Debug,eqstate=eqstate).mld_z

    def pe_anomaly(self,gradient=False,Dpt=0):
        if gradient:
            print('not ready for gradient in column mode')
            return -999
        else:
            dprhodz = self.prho*0.0
            
        return _pe_anomaly(self.prho,dprhodz,self.zc,self.dz,DPT=Dpt).PE
    
                  
                  
                  
                  
                  
                  
                  
