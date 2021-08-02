"""
Contains the codes needed to compute the PE anomaly associated with mixing a water column to a specified depth.
"""

import numpy as np

def PE_Kernel_linear(R_layer,dRdz,Z_U,Z_L):
    PE =  ( R_layer/2.*(Z_U**2-Z_L**2) +
            dRdz/3.*(Z_U**3-Z_L**3) -
            dRdz*(Z_U+Z_L)/4.*(Z_U**2-Z_L**2) )
    return PE

def PE_Kernel(R_layer,Z_U,Z_L):
    PE =  ( R_layer/2.*(Z_U**2-Z_L**2))
    return PE

def MixLayers(Tracer,dZ,nlev):
    DZ_Mixed = np.sum(dZ[:nlev,...],axis=0)
    T_Mixed = np.sum((Tracer*dZ)[:nlev,...],axis=0)/DZ_Mixed
    return T_Mixed, DZ_Mixed

class pe_anomaly():
    def __init__(self,Rho0_layer,dRho0dz_layer,Zc,dZi,DPT):
        self.compute(Rho0_layer,dRho0dz_layer,Zc,dZi,DPT)

    def compute(self,Rho0_layer,dRho0dz_layer,Zc,dZi,DPT):
    
        dZ = np.copy(dZi)
        
        ND = Rho0_layer.shape[1:]
        NZ = Rho0_layer.shape[0]
        
        if np.size(DPT)<=1:
            DPT = np.broadcast_to(DPT,ND)
            
        Z_U = Zc+dZ/2.
        Z_L = Zc-dZ/2.
        
        Rho0_Mixed = np.copy(Rho0_layer)
        dz_Mixed = dZ[0]
        
        PEdelta = np.zeros(ND)+np.NaN
        
        ACTIVE = np.ones(ND,dtype='bool')
        ACTIVE[np.isnan(Rho0_layer[0,...])] = False
        
        FINAL = np.zeros(ND,dtype='bool')
        
        z=-1
        while (z<NZ-1 and np.sum(ACTIVE)>0):
            z += 1
            
            FINAL[ACTIVE] =  (Z_L[z,ACTIVE]<DPT[ACTIVE])
            #Cut the bottom of the cell
            Z_L[z,FINAL] = DPT[FINAL]
            #Update the dZ
            dZ[z,FINAL] = Z_U[z,FINAL]-Z_L[z,FINAL]
            
            Rho0_layers_linear = Rho0_layer[:z+1,FINAL]
            Rho0_layers_linear[z,:] = (
                (Rho0_layer[z,FINAL] + dRho0dz_layer[z,FINAL] * (0.5*dZi[z,FINAL]) ) #value at top                                 
                - 0.5 * dRho0dz_layer[z,FINAL] * dZ[z,FINAL] #adjustment from value at top to value at center                         
            )
            
            PE_before  = np.sum(PE_Kernel_linear(Rho0_layers_linear,
                                                 dRho0dz_layer[:z+1,FINAL],
                                                 Z_U[:z+1,FINAL],
                                                 Z_L[:z+1,FINAL]),
                                axis=0)
            
            Rho0_Mixed[:z+1,FINAL],dz_Mixed[FINAL] = MixLayers(Rho0_layers_linear,
                                                               dZ[:z+1,FINAL],
                                                               z+1)
            PE_after  = np.sum(PE_Kernel(Rho0_Mixed[:z+1,FINAL],
                                         Z_U[:z+1,FINAL],
                                         Z_L[:z+1,FINAL]),
                               axis=0)
            
            PEdelta[FINAL] = PE_after - PE_before
            
            ACTIVE[FINAL] = False
            FINAL[:] = False
        
        self.PE = PEdelta
