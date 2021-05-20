import numpy as np
import gsw as gsw
import warnings

def PE_Kernel_linear(R_layer,dRdz,Z_U,Z_C,Z_L):
    PE =  ( R_layer/2.*(Z_U**2-Z_L**2) +
            dRdz/3.*(Z_U**3-Z_L**3) -
            dRdz*Z_C/2*(Z_U**2-Z_L**2) )
    return PE

def PE_Kernel_linear_new(R_layer,dRdz,Z_U,Z_L):
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

class energy():
    """
    A class storing the energy based computation methods
    """
    
    def __init__():
        pass
    
    def energetic_mixing_depth_TS_nd(T_layer,S_layer,P_layer,Zc,dZ,Energy,Debug=False):

        ND = T_layer.shape[1:]
        NZ = T_layer.shape[0]
        
        Z_U = Zc+dZ/2.
        Z_L = Zc-dZ/2.
        
        T_Mixed = np.copy(T_layer)
        S_Mixed = np.copy(S_layer)
        dz_Mixed = dZ[0]
        
        Rho_Unmixed = gsw.density.rho(S_layer,T_layer,P_layer)
        
        ACTIVE = np.ones(ND,dtype='bool')
        FINISHED = np.zeros(ND,dtype='bool')
        
        ACTIVE[np.isnan(T_layer[0,...])|np.isnan(S_layer[0,...])] = False
        FINISHED[np.isnan(T_layer[0,...])|np.isnan(S_layer[0,...])] = True
        
        MLD              = np.add(np.NaN,np.array(np.zeros(ND)))
        MLD[ACTIVE]      = 0.0
        PE_after              = np.add(np.NaN,np.array(np.zeros(ND)))
        PE_after[ACTIVE]      = 0.0
        PE_before              = np.add(np.NaN,np.array(np.zeros(ND)))
        PE_before[ACTIVE]      = 0.0
        PE_before_above              = np.add(np.NaN,np.array(np.zeros(ND)))
        PE_before_above[ACTIVE]      = 0.0
        dz_Mixed              = np.add(np.NaN,np.array(np.zeros(ND)))
        dz_Mixed[ACTIVE]      = 0.0
        IT_total         = np.add(np.NaN,np.array(np.zeros(ND)))
        IT_total[ACTIVE] = 0
        
        z=-1
        while(z<NZ-1 and np.sum(ACTIVE)>0):
            z += 1
            
            CNVG = np.zeros(ND,dtype='bool')
            ACTIVE[np.isnan(T_layer[z,...])|np.isnan(S_layer[z,...])] = False
            FINISHED[np.isnan(T_layer[z,...])|np.isnan(S_layer[z,...])] = True
            FINAL = np.zeros(ND,dtype='bool')
            
            PE_before_above[ACTIVE] = np.sum(PE_Kernel(Rho_Unmixed[:z,ACTIVE],
                                                       Z_U[:z,ACTIVE],
                                                       Z_L[:z,ACTIVE]),
                                             axis=0)
            PE_before[ACTIVE]  = np.sum(PE_Kernel(Rho_Unmixed[:z+1,ACTIVE],
                                                  Z_U[:z+1,ACTIVE],
                                                  Z_L[:z+1,ACTIVE]),
                                        axis=0)
            
            T_Mixed[:z+1,ACTIVE],dz_Mixed[ACTIVE] = MixLayers(T_layer[:,ACTIVE],
                                                              dZ[:,ACTIVE],
                                                              z+1)
            S_Mixed[:z+1,ACTIVE],dz_Mixed[ACTIVE] = MixLayers(S_layer[:,ACTIVE],
                                                              dZ[:,ACTIVE],
                                                              z+1)
            
            Rho = gsw.density.rho(S_Mixed[:z+1,ACTIVE],
                                  T_Mixed[:z+1,ACTIVE],
                                  P_layer[:z+1,ACTIVE])
            
            PE_after[ACTIVE]  = np.sum(PE_Kernel(Rho[:z+1,...],
                                                 Z_U[:z+1,ACTIVE],
                                                 Z_L[:z+1,ACTIVE])
                                       ,axis=0)
            
            FINAL[ACTIVE] = (PE_after[ACTIVE]-PE_before[ACTIVE]>Energy)
            ACTIVE[ACTIVE] = (PE_after[ACTIVE]-PE_before[ACTIVE]<Energy)
            
            ITCOUNT = np.zeros(np.sum(FINAL))
            
            MLD[ACTIVE] += dZ[z,ACTIVE]
            
            IT=-1
            #print(z,np.sum(FINAL),np.sum(FINISHED))
            DZlo = (np.zeros(np.sum(FINAL)))
            DZup = (dZ[z,FINAL])
            
            while (np.sum(FINAL)>=1):
                IT+=1
                
                DZ = np.vstack((dZ[:z,FINAL],0.5*(DZlo+DZup)))
                
                PE_before[FINAL] = PE_before_above[FINAL] + PE_Kernel(Rho_Unmixed[z,FINAL],
                                                                      Z_U[z,FINAL],
                                                                      Z_U[z,FINAL]-DZ[z,:])
                T_Mixed[:z+1,FINAL],dz_Mixed[FINAL] = MixLayers(T_layer[:z+1,FINAL],
                                                                DZ,
                                                                z+1)
                S_Mixed[:z+1,FINAL],dz_Mixed[FINAL] = MixLayers(S_layer[:z+1,FINAL],
                                                                DZ,
                                                                z+1)
                Rho = gsw.density.rho(S_Mixed[:,FINAL],
                                      T_Mixed[:,FINAL],
                                      P_layer[:,FINAL])
                PE_after[FINAL]  = np.sum(PE_Kernel(Rho[:z+1,:],
                                                    Z_U[:z+1,FINAL],
                                                    Z_U[:z+1,FINAL]-DZ),
                                          axis=0)
                
                if (IT<100):
                    CNVG[FINAL] = abs(PE_after[FINAL]-PE_before[FINAL]-Energy)<Energy*1.e-5
                    cf = abs(PE_after[FINAL]-PE_before[FINAL]-Energy)<Energy*1.e-5
                    
                    MLD[CNVG] = dz_Mixed[CNVG]
                    FINISHED[CNVG] = True
                    FINAL[CNVG] = False
                    
                    TOO_HIGH = (PE_after[FINAL]-PE_before[FINAL])>Energy
                    TOO_LOW  =  (PE_after[FINAL]-PE_before[FINAL])<Energy
                    DZup = DZup[~cf]
                    DZlo = DZlo[~cf]
                    DZup[TOO_HIGH]=DZ[-1,~cf][TOO_HIGH]
                    DZlo[TOO_LOW]=DZ[-1,~cf][TOO_LOW]
                    
                    
                else:
                    print(IT,np.sum(FINAL),' #not converged')
                    print(abs(PE_after[FINAL]-PE_before[FINAL]-Energy))
                    print(dz_Mixed[FINAL])
                    print(DZup)
                    print(DZlo)
                    print(T_layer[:z+1,FINAL][:,0])
                    print(S_layer[:z+1,FINAL][:,0])
                    print(S_layer[:z+1,FINAL][:,0])
                    print(P_layer[:z+1,FINAL][:,0])
                    print(Zc[:z+1,FINAL][:,0])
                    print(dZ[:z+1,FINAL][:,0])
                    print(Energy)
                    MLD[FINAL] = dz_Mixed[FINAL]
                    FINAL[FINAL] = False
                    asdf
                    
                    
                    
        return MLD
                    
    def energetic_mixing_depth_Rho0_nd(Rho0_layer,Zc,dZ,Energy,Debug=False):
        
        ND = Rho0_layer.shape[1:]
        NZ = Rho0_layer.shape[0]
        
        Z_U = Zc+0.5*dZ
        Z_L = Zc-0.5*dZ
        
        Rho0_Mixed = np.copy(Rho0_layer)
        dz_Mixed = dZ[0]
        
        ACTIVE = np.ones(ND,dtype='bool')
        FINISHED = np.zeros(ND,dtype='bool')
        
        ACTIVE[np.isnan(Rho0_layer[0,...])] = False
        FINISHED[np.isnan(Rho0_layer[0,...])] = True
        
        MLD              = np.add(np.NaN,np.array(np.zeros(ND)))
        MLD[ACTIVE]      = 0.0
        PE_after              = np.add(np.NaN,np.array(np.zeros(ND)))
        PE_after[ACTIVE]      = 0.0
        PE_before              = np.add(np.NaN,np.array(np.zeros(ND)))
        PE_before[ACTIVE]      = 0.0
        PE_before_above              = np.add(np.NaN,np.array(np.zeros(ND)))
        PE_before_above[ACTIVE]      = 0.0
        dz_Mixed              = np.add(np.NaN,np.array(np.zeros(ND)))
        dz_Mixed[ACTIVE]      = 0.0
        IT_total         = np.add(np.NaN,np.array(np.zeros(ND)))
        IT_total[ACTIVE] = 0
        
        z=-1
        while(z<NZ-1 and np.sum(ACTIVE)>0):
            z += 1
            
            CNVG = np.zeros(ND,dtype='bool')
            ACTIVE[np.isnan(Rho0_layer[z,...])] = False
            FINISHED[np.isnan(Rho0_layer[z,...])] = True
            FINAL = np.zeros(ND,dtype='bool')
            
            PE_before_above[ACTIVE] = np.sum(PE_Kernel(Rho0_layer[:z,ACTIVE],
                                                       Z_U[:z,ACTIVE],
                                                       Z_L[:z,ACTIVE]),
                                             axis=0)
            PE_before[ACTIVE]  = np.sum(PE_Kernel(Rho0_layer[:z+1,ACTIVE],
                                                  Z_U[:z+1,ACTIVE],
                                                  Z_L[:z+1,ACTIVE]),
                                        axis=0)
            
            Rho0_Mixed[:z+1,ACTIVE],dz_Mixed[ACTIVE] = MixLayers(Rho0_layer[:,ACTIVE],
                                                                 dZ[:,ACTIVE],
                                                                 z+1)
            
            PE_after[ACTIVE]  = np.sum(PE_Kernel(Rho0_Mixed[:z+1,ACTIVE],
                                                 Z_U[:z+1,ACTIVE],
                                                 Z_L[:z+1,ACTIVE])
                                       ,axis=0)
            
            #        print(z,Rho0_Mixed[0,ACTIVE],dz_Mixed[ACTIVE])#PE_after[ACTIVE],PE_before[ACTIVE],PE_after[ACTIVE]-PE_before[ACTIVE],Energy)
            FINAL[ACTIVE] = (PE_after[ACTIVE]-PE_before[ACTIVE]>=Energy)
            ACTIVE[ACTIVE] = (PE_after[ACTIVE]-PE_before[ACTIVE]<Energy)

            ITCOUNT = np.zeros(np.sum(FINAL))

            MLD[ACTIVE] += dZ[z,ACTIVE]
            
            IT=-1
            #print(z,np.sum(FINAL),np.sum(FINISHED))
            DZlo = (np.zeros(np.sum(FINAL)))
            DZup = (dZ[z,FINAL])
            while (np.sum(FINAL)>=1):
                IT+=1
                
                DZ = np.vstack((dZ[:z,FINAL],0.5*(DZlo+DZup)))
                
                PE_before[FINAL] = PE_before_above[FINAL] + PE_Kernel(Rho0_layer[z,FINAL],
                                                                      Z_U[z,FINAL],
                                                                      Z_U[z,FINAL]-DZ[z,:])
                Rho0_Mixed[:z+1,FINAL],dz_Mixed[FINAL] = MixLayers(Rho0_layer[:z+1,FINAL],
                                                                   DZ,
                                                                   z+1)
                PE_after[FINAL]  = np.sum(PE_Kernel(Rho0_Mixed[:z+1,FINAL],
                                                    Z_U[:z+1,FINAL],
                                                    Z_U[:z+1,FINAL]-DZ),
                                      axis=0)
                
                if (IT<50):
                    #                print(IT,DZ[-1],DZlo,DZup,Rho0_Mixed[0,FINAL],dz_Mixed[FINAL])
                    #                print(PE_after[FINAL],PE_before[FINAL],PE_after[FINAL]-PE_before[FINAL]-Energy)
                    CNVG[FINAL] = abs(PE_after[FINAL]-PE_before[FINAL]-Energy)<Energy*1.e-5
                    cf = abs(PE_after[FINAL]-PE_before[FINAL]-Energy)<Energy*1.e-5
                    
                    MLD[CNVG] = dz_Mixed[CNVG]
                    FINISHED[CNVG] = True
                    FINAL[CNVG] = False
                    
                    TOO_HIGH = (PE_after[FINAL]-PE_before[FINAL])>Energy
                    TOO_LOW  =  (PE_after[FINAL]-PE_before[FINAL])<Energy
                    DZup = DZup[~cf]
                    DZlo = DZlo[~cf]
                    DZup[TOO_HIGH]=DZ[-1,~cf][TOO_HIGH]
                    DZlo[TOO_LOW]=DZ[-1,~cf][TOO_LOW]
                    

                else:
                    print(IT,np.sum(FINAL),' #not converged')
                    print(PE_after[FINAL],PE_before[FINAL],PE_after[FINAL]-PE_before[FINAL],Energy)
                    print(DZup,DZlo,DZ[-1,:])
                    print(Z_U[:,FINAL].T)
                    print(Z_L[:,FINAL].T)
                    print(Rho0_layer[:,FINAL].T)
                    print(Zc[:,FINAL].T)
                    print(dZ[:,FINAL].T)
                    MLD[FINAL] = dz_Mixed[FINAL]
                    FINAL[FINAL] = False
                    asdf

        return MLD

    def energetic_mixing_depth_Rho0_Linear_nd(Rho0_layer,dRho0dz_layer,Zc,dZ,Energy,CNVG_T=1.e-5,Debug=False):
        
        # The syntax below is written assuming an nd structure of Rho0, dRho0dz, Zc, and dZ, where n is >=2.
        # If a single column is passed in we convert to a 2d array.
        Rho0_layer = np.atleast_2d(Rho0_layer).T
        dRho0dz_layer = np.atleast_2d(dRho0dz_layer).T
        Zc = np.broadcast_to(Zc,np.shape(Rho0_layer.T)).T
        dZ = np.broadcast_to(dZ,np.shape(Rho0_layer.T)).T

        ND = Rho0_layer.shape[1:]
        NZ = Rho0_layer.shape[0]
        
        Z_U = Zc+dZ/2.
        Z_L = Zc-dZ/2.
        
        Rho0_Mixed = np.copy(Rho0_layer)
        dz_Mixed = dZ[0]
        
        ACTIVE = np.ones(ND,dtype='bool')
        FINISHED = np.zeros(ND,dtype='bool')
        
        ACTIVE[np.isnan(Rho0_layer[0,...])] = False
        FINISHED[np.isnan(Rho0_layer[0,...])] = True
        
        MLD              = np.add(np.NaN,np.array(np.zeros(ND)))
        MLD[ACTIVE]      = 0.0
        PE_after              = np.add(np.NaN,np.array(np.zeros(ND)))
        PE_after[ACTIVE]      = 0.0
        PE_before              = np.add(np.NaN,np.array(np.zeros(ND)))
        PE_before[ACTIVE]      = 0.0
        PE_before_above              = np.add(np.NaN,np.array(np.zeros(ND)))
        PE_before_above[ACTIVE]      = 0.0
        dz_Mixed              = np.add(np.NaN,np.array(np.zeros(ND)))
        dz_Mixed[ACTIVE]      = 0.0
        IT_total         = np.add(np.NaN,np.array(np.zeros(ND)))
        IT_total[ACTIVE] = 0
        
        z=-1
        while(z<NZ-1 and np.sum(ACTIVE)>0):
            z += 1
            
            CNVG = np.zeros(ND,dtype='bool')
            ACTIVE[np.isnan(Rho0_layer[z,...])] = False
            FINISHED[np.isnan(Rho0_layer[z,...])] = True
            FINAL = np.zeros(ND,dtype='bool')
            
            PE_before_above[ACTIVE] = (
                np.sum(PE_Kernel_linear_new(Rho0_layer[:z,ACTIVE],
                                            dRho0dz_layer[:z,ACTIVE],
                                            Z_U[:z,ACTIVE],
                                            Z_L[:z,ACTIVE]), axis=0)
            )
            
            PE_before[ACTIVE]  = (
                np.sum(PE_Kernel_linear_new(Rho0_layer[:z+1,ACTIVE],
                                            dRho0dz_layer[:z+1,ACTIVE],
                                            Z_U[:z+1,ACTIVE],
                                            Z_L[:z+1,ACTIVE]), axis=0)
            )
            
            Rho0_Mixed[:z+1,ACTIVE],dz_Mixed[ACTIVE] = (
                MixLayers(Rho0_layer[:,ACTIVE],
                          dZ[:,ACTIVE],
                          z+1)
            )
            
            PE_after[ACTIVE]  = (
                np.sum(PE_Kernel(Rho0_Mixed[:z+1,ACTIVE],
                                 Z_U[:z+1,ACTIVE],
                                 Z_L[:z+1,ACTIVE])
                       ,axis=0)
            )
            
            FINAL[ACTIVE] = (PE_after[ACTIVE]-PE_before[ACTIVE]>=Energy)
            ACTIVE[ACTIVE] = (PE_after[ACTIVE]-PE_before[ACTIVE]<Energy)
            
            ITCOUNT = np.zeros(np.sum(FINAL))
            
            MLD[ACTIVE] += dZ[z,ACTIVE]
            
            IT=-1
            #print(z,np.sum(FINAL),np.sum(FINISHED))
            DZlo = (np.zeros(np.sum(FINAL)))
            DZup = (dZ[z,FINAL])
            
            while (np.sum(FINAL)>=1):
                IT+=1
                
                DZ = np.vstack((dZ[:z,FINAL],0.5*(DZlo+DZup)))
                
                Rho0_layers_linear = Rho0_layer[:z+1,FINAL]
                Rho0_layers_linear[z,:] = (
                    (Rho0_layer[z,FINAL] + dRho0dz_layer[z,FINAL] * (0.5*dZ[z,FINAL]) ) #value at top
                    - 0.5 * dRho0dz_layer[z,FINAL] * DZ[z,:] #adjustment from value at top to value at center
                )
                
                PE_before[FINAL] = (
                    PE_before_above[FINAL] 
                    + PE_Kernel_linear_new(Rho0_layers_linear[z,:],
                                           dRho0dz_layer[z,FINAL],
                                           Z_U[z,FINAL],
                                           Z_U[z,FINAL]-DZ[z,:])
                )
                
                Rho0_Mixed[:z+1,FINAL],dz_Mixed[FINAL] = (
                    MixLayers(Rho0_layers_linear, DZ, z+1)
                )
                
                PE_after[FINAL]  = (
                    np.sum(PE_Kernel(Rho0_Mixed[:z+1,FINAL],
                                     Z_U[:z+1,FINAL],
                                     Z_U[:z+1,FINAL]-DZ),
                           axis=0)
                )
                
                if (IT<50):
                    CNVG[FINAL] = abs(PE_after[FINAL]-PE_before[FINAL]-Energy)<=(Energy*CNVG_T)
                    cf = abs(PE_after[FINAL]-PE_before[FINAL]-Energy)<=(Energy*CNVG_T)
                    
                    MLD[CNVG] = dz_Mixed[CNVG]
                    FINISHED[CNVG] = True
                    FINAL[CNVG] = False
                    
                    TOO_HIGH = (PE_after[FINAL]-PE_before[FINAL])>Energy
                    TOO_LOW  =  (PE_after[FINAL]-PE_before[FINAL])<Energy
                    DZup = DZup[~cf]
                    DZlo = DZlo[~cf]
                    DZup[TOO_HIGH]=DZ[-1,~cf][TOO_HIGH]
                    DZlo[TOO_LOW]=DZ[-1,~cf][TOO_LOW]
                    

                else:
                    print(IT,np.sum(FINAL),' #not converged')
                    print(PE_after[FINAL],PE_before[FINAL],PE_after[FINAL]-PE_before[FINAL],Energy)
                    print(DZup,DZlo,DZ[-1,:])
                    print(Z_U[:,FINAL])
                    MLD[FINAL] = dz_Mixed[FINAL]
                    FINAL[FINAL] = False
                    asdf
                    
        return MLD
