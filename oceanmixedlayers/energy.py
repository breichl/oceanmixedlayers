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

def PE_Kernel_dP1(Zc,dP):
    return Zc*(-dP)

def PE_Kernel_dP(Rhoxg,Pt,Pb):
    return (-1./(2*Rhoxg)*(Pt**2-Pb**2))

def PE_Kernel(R_layer,Z_U,Z_L):
    PE =  ( R_layer/2.*(Z_U**2-Z_L**2))
    return PE

def MixLayers(Tracer,dZ,nlev):
    DZ_Mixed = np.sum(dZ[:nlev,...],axis=0)
    T_Mixed = np.sum((Tracer*dZ)[:nlev,...],axis=0)/DZ_Mixed
    return T_Mixed, DZ_Mixed

class mld_delta_pe():
    """
    A class storing the energy based computation method from Temperature and Salinity
    The inputs needed are:
    T_layer - temperature of the layer
    S_layer - salinity of the layer
    Vc - vertical coordinate (depth or pressure)
    dV - "thickness" of coordinate (in depth or pressure)
    """
    
    def __init__(self,
                 #Inputs
                 Vc,dV,T_layer,S_layer,energy,
                 #coordinate option
                 coord='pressure',
                 #Debugging
                 Debug=False,
                 #Constants that can be set
                 grav=9.81,rho0=1025.):
        self.grav = grav
        self.rho0 = rho0

        # The syntax below is written assuming an nd structure of T, S, Zc, and dZ, where n is >=2.
        # If a single column is passed in we convert to a 2d array.
        if (len(np.shape(T_layer))==1):
            T_layer = np.atleast_2d(T_layer).T
            S_layer = np.atleast_2d(S_layer).T
        if (np.shape(T_layer)!=np.shape(Vc)):
            Vc = np.broadcast_to(Vc,np.shape(T_layer.T)).T
            dV = np.broadcast_to(dV,np.shape(T_layer.T)).T

        # Depending on the the coordinate we need to compute P or Z and density
        # Since density is a function of pressure we may need to iterate between computing
        # density and pressure.  A simplified approach is to use the hydrostatic pressure
        # to estimate the density, and then compute the pressure from the density.
        if coord=='pressure':
            #Pressure will be constant because we will evaluate in pressure coordinate
            Pc = Vc
            dP = dV
            #Sets Rho_i
            Rho_i = gsw.density.rho(S_layer,T_layer,Pc/1.e4)

            #Compute the layer thicknesses 
            dZ_i = -dP/(self.grav*Rho_i)
            Zc_i = np.zeros(np.shape(dZ_i))
            Zc_i[0,...]=-0.5*dZ_i[0,...]
            for zz in range(1,np.shape(dV)[0]):
                Zc_i[zz,...]=Zc_i[zz-1,...]-dZ_i[zz-1,...]*0.5-dZ_i[zz,...]*0.5

        else:
            #Find the hydrostatic pressure
            #Zc_i is the input Vc
            Zc_i = Vc
            #dZ_i is the input dV
            dZ_i = dV
            #Pressure needs to be calculated and will be help constant bc we will 
            #evaluate in pressure coordinate
            #Guess Hydrostatic
            Pi = np.array([0.,]+list(np.cumsum(dZ_i*self.grav*self.rho0)))
            Pc = 0.5*(Pi[1:]+Pi[:-1])
            #Compute Rho_i assuming hydrostatic pressure
            Rho_i = gsw.density.rho(S_layer,T_layer,Pc/1.e4)
            #Iterate to make density consistent w/ pressure if Coord is not 'hydrostatic'
            if coord=='depth':
                #Update the density and pressure iteratively to convergence
                for ii in range(20):
                    #How many iterations?
                    Pi = np.array([0.,]+list(np.cumsum(dZ_i*self.grav*Rho_i)))
                    Pc = 0.5*(Pi[1:]+Pi[:-1])
                    Rho_0 = gsw.density.rho(S_layer,T_layer,Pc/1.e4)
            elif coord=='hydrostatic':
                #Don't update the density
                pass
            else:
                print('Something wrong here...')
                return
            #Update the pressure to be consistent with the density
            #Pressure will not be updated from here on
            Pi = np.array([0.,]+list(np.cumsum(dZ*self.grav*Rho_i)))
            Pc = 0.5*(Pi[1:]+Pi[:-1])
            dP = Pi[:-1]-Pi[1:] #(this will be negative)
        
        self.compute_MLD(T_layer,
                         S_layer,
                         Rho_i,
                         Pc,
                         dP,
                         Zc_i,
                         dZ_i,
                         energy,
                         Debug=Debug)

    def compute_MLD(self,T_i,S_i,Rho_i,Pc,dP,Zc_i,dZ_i,energy,Debug=False):

        ND = T_i.shape[1:]
        NZ = T_i.shape[0]
        
        T_x = np.copy(T_i)
        S_x = np.copy(S_i)
        dP_x = np.copy(dP[0])

        # Initial upper/lower interfaces of each cell in position
        Z_U_i = Zc_i+dZ_i/2.
        Z_L_i = Zc_i-dZ_i/2.
        P_U = Pc+dP/2.
        P_L = Pc-dP/2.

        # We now proceed to iterate down the column until we find the depth
        # where the mixing satisfies the supplied energy

        ACTIVE = np.ones(ND,dtype='bool')
        FINISHED = np.zeros(ND,dtype='bool')
        
        ACTIVE[np.isnan(T_i[0,...])|np.isnan(S_i[0,...])] = False
        FINISHED[np.isnan(T_i[0,...])|np.isnan(S_i[0,...])] = True
        
        MLD_z                     = np.add(np.NaN,np.array(np.zeros(ND)))
        MLD_z[ACTIVE]             = 0.0
        MLD_p                     = np.add(np.NaN,np.array(np.zeros(ND)))
        MLD_p[ACTIVE]             = 0.0
        PE_x                = np.add(np.NaN,np.array(np.zeros(ND)))
        PE_x[ACTIVE]        = 0.0
        PE_i               = np.add(np.NaN,np.array(np.zeros(ND)))
        PE_i[ACTIVE]       = 0.0
        PE_i_above         = np.add(np.NaN,np.array(np.zeros(ND)))
        PE_i_above[ACTIVE] = 0.0
        MLDp                = np.add(np.NaN,np.array(np.zeros(ND)))
        MLDp[ACTIVE]        = 0.0
        MLDz                = np.add(np.NaN,np.array(np.zeros(ND)))
        MLDz[ACTIVE]        = 0.0
        IT_total                = np.add(np.NaN,np.array(np.zeros(ND)))
        IT_total[ACTIVE]        = 0
        
        z=-1
        while(z<NZ-1 and np.sum(ACTIVE)>0):
            z += 1
            
            CNVG = np.zeros(ND,dtype='bool')
            ACTIVE[np.isnan(T_i[z,...])|np.isnan(S_i[z,...])] = False
            FINISHED[np.isnan(T_i[z,...])|np.isnan(S_i[z,...])] = True
            FINAL = np.zeros(ND,dtype='bool')
            
            #PE_i_above[ACTIVE] = np.sum(PE_Kernel_dP(Zc_i[:z,ACTIVE],
            #                                         dP[:z,ACTIVE]),
            #                            axis=0)
            PE_i_above[ACTIVE] = np.sum(PE_Kernel_dP(self.grav*Rho_i[:z,ACTIVE],
                                                     P_U[:z,ACTIVE],
                                                     P_L[:z,ACTIVE]),
                                        axis=0)
            #PE_i[ACTIVE] = np.sum(PE_Kernel_dP(Zc_i[:z+1,ACTIVE],
            #                                   dP[:z+1,ACTIVE]),
            #                      axis=0)
            PE_i[ACTIVE] = np.sum(PE_Kernel_dP(self.grav*Rho_i[:z+1,ACTIVE],
                                                     P_U[:z+1,ACTIVE],
                                                     P_L[:z+1,ACTIVE]),
                                        axis=0)
        
            # We need to mix in pressure!
            T_x[:z+1,ACTIVE],_ = MixLayers(T_i[:,ACTIVE],
                                           -dP[:,ACTIVE],
                                           z+1)
            S_x[:z+1,ACTIVE],_ = MixLayers(S_i[:,ACTIVE],
                                           -dP[:,ACTIVE],
                                           z+1)
            
            Rho_x = gsw.density.rho(S_x[:z+1,ACTIVE],
                                    T_x[:z+1,ACTIVE],
                                    Pc[:z+1,ACTIVE]/1.e4)
            
            #Recompute the layer thicknesses 
            dZ_x = -dP[:z+1,ACTIVE]/(self.grav*Rho_x)

            #Update Zc
            Zc_x = np.zeros(np.shape(dZ_x))
            Zc_x[0,...]=-0.5*dZ_x[0,...]
            if z>0:
                for zz in range(1,z+1):
                    Zc_x[zz,...]=Zc_x[zz-1,...]-dZ_x[zz-1,...]*0.5-dZ_x[zz,...]*0.5


            #PE_x[ACTIVE]  = np.sum(PE_Kernel_dP(Zc_x,
            #                            dP[:z+1,ACTIVE])
            #               ,axis=0)
            PE_x[ACTIVE]  = np.sum(PE_Kernel_dP(self.grav*Rho_x,
                                                P_U[:z+1,ACTIVE],
                                                P_L[:z+1,ACTIVE],
                                            )
                                   ,axis=0)
            
            FINAL[ACTIVE] = (PE_x[ACTIVE]-PE_i[ACTIVE]>energy)
            ACTIVE[ACTIVE] = (PE_x[ACTIVE]-PE_i[ACTIVE]<energy)
            
            ITCOUNT = np.zeros(np.sum(FINAL))
            
            # MLD is computed in pressure, can be converted to depth later
            MLD_p[ACTIVE] -= dP[z,ACTIVE]
            
            IT=-1
            #print(z,np.sum(FINAL),np.sum(FINISHED))

            #Upper and lower bounds on dP for iteration
            DPlo = (np.zeros(np.sum(FINAL)))
            DPup = (dP[z,FINAL])
            if Debug: print(PE_i,PE_x,PE_x-PE_i)

            while (np.sum(FINAL)>=1):
                # Enter this loop if any points exceeded the energy provided.
                # - now find the MLD within the layer.
                IT+=1
                
                DP = np.vstack((dP[:z,FINAL],0.5*(DPlo+DPup)))
                PC = Pc[:z+1,FINAL]
                PC[z,...] = Pc[z-1,FINAL]-0.5*(dP[z-1,FINAL]+DP[z,...])


                #Recompute the reduced bottom layer thickness for the initial state
                Rho_i_bot = gsw.density.rho(S_i[z,FINAL],T_i[z,FINAL],PC[z,...]/1.e4)
                #PE_i[FINAL] = PE_i_above[FINAL] + PE_Kernel_dP(Zc_i[z,FINAL],DP[z])
                PE_i[FINAL] = PE_i_above[FINAL] + PE_Kernel_dP(self.grav*Rho_i_bot,
                                                               P_U[z,FINAL],P_U[z,FINAL]-DP[z])
                
                T_x[:z+1,FINAL],MLDp[FINAL] = MixLayers(T_i[:z+1,FINAL],
                                                 -DP,
                                                 z+1)
                S_x[:z+1,FINAL],MLDp[FINAL] = MixLayers(S_i[:z+1,FINAL],
                                                 -DP,
                                                 z+1)
                Rho_x = gsw.density.rho(S_x[:z+1,FINAL],
                                        T_x[:z+1,FINAL],
                                        PC/1.e4)

                #Recompute the layer thicknesses 
                dZ_x = -DP/(self.grav*Rho_x)
                MLDz[FINAL] = np.sum(dZ_x,axis=0)

                #Update Zc
                Zc_x = np.zeros(np.shape(dZ_x))
                Zc_x[0,...]=-0.5*dZ_x[0,...]
                if z>0:
                    for zz in range(1,z+1):
                        Zc_x[zz,...]=Zc_x[zz-1,...]-dZ_x[zz-1,...]*0.5-dZ_x[zz,...]*0.5

                PE_x[FINAL]  = np.sum(PE_Kernel_dP(self.grav*Rho_x,P_U[:z+1,FINAL],P_U[:z+1,FINAL]-DP),axis=0)

                if Debug: print(PE_i,PE_x,PE_x-PE_i)

                if (IT<100):
                    CNVG[FINAL] = abs(PE_x[FINAL]-PE_i[FINAL]-energy)<energy*1.e-5
                    cf = abs(PE_x[FINAL]-PE_i[FINAL]-energy)<energy*1.e-5
                    
                    MLD_p[CNVG] = MLDp[CNVG]
                    MLD_z[CNVG] = MLDz[CNVG]
                    FINISHED[CNVG] = True
                    FINAL[CNVG] = False
                    
                    TOO_HIGH = (PE_x[FINAL]-PE_i[FINAL])>energy
                    TOO_LOW  =  (PE_x[FINAL]-PE_i[FINAL])<energy
                    DPup = DPup[~cf]
                    DPlo = DPlo[~cf]
                    DPup[TOO_HIGH]=DP[-1,~cf][TOO_HIGH]
                    DPlo[TOO_LOW]=DP[-1,~cf][TOO_LOW]
                    
                    
                else:
                    # These are definitely wrong...
                    print(IT,np.sum(FINAL),' #not converged')
                    print(abs(PE_after[FINAL]-PE_before[FINAL]-energy))
                    print(dz_Mixed[FINAL])
                    print(DZup)
                    print(DZlo)
                    print(T_i[:z+1,FINAL][:,0])
                    print(S_i[:z+1,FINAL][:,0])
                    print(S_i[:z+1,FINAL][:,0])
                    print(Pc[:z+1,FINAL][:,0])
                    print(Zc[:z+1,FINAL][:,0])
                    print(dZ[:z+1,FINAL][:,0])
                    print(energy)
                    MLD[FINAL] = dz_Mixed[FINAL]
                    FINAL[FINAL] = False
                    asdf
                  
        self.mld_p = MLD_p/1.e4
        self.mld_z = MLD_z

class mld_pe_anomaly():
    """
    A class storing the energy based computation methods
    """
    
    def __init__(self,
                 #Inputs
                 Zc, dZ, Rho0_layer,
                 #Optional argument
                 dRho0dz_layer=[],
                 #Arguments with default values
                 energy=10,CNVG_T=1.e-5,Debug=False,grav=9.81,rho0=1025.):
        self.grav = grav
        self.rho0 = rho0
        if(np.size(dRho0dz_layer)==0): dRho0dz_layer = Rho0_layer*0.0
        self.compute_MLD(Zc, dZ, Rho0_layer, dRho0dz_layer, energy, CNVG_T, Debug)
        
    def compute_MLD(self,Zc,dZ,Rho0_layer,dRho0dz_layer,energy,CNVG_T=1.e-5,Debug=False):

        energy = energy/self.grav

        # The syntax below is written assuming an nd structure of Rho0, dRho0dz, Zc, and dZ, where n is >=2.
        # If a single column is passed in we convert to a 2d array.
        if (len(np.shape(Rho0_layer))==1):
            Rho0_layer = np.atleast_2d(Rho0_layer).T
            dRho0dz_layer = np.atleast_2d(dRho0dz_layer).T
        if (np.shape(Rho0_layer)!=np.shape(Zc)):
            Zc = np.broadcast_to(Zc,np.shape(Rho0_layer.T)).T
            dZ = np.broadcast_to(dZ,np.shape(Rho0_layer.T)).T
        
        ND = Rho0_layer.shape[1:]
        NZ = Rho0_layer.shape[0]

        Z_U = Zc+dZ/2.
        Z_L = Zc-dZ/2.
        
        Rho0_Mixed = np.copy(Rho0_layer)
        
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
            
            FINAL[ACTIVE] = (PE_after[ACTIVE]-PE_before[ACTIVE]>=energy)
            ACTIVE[ACTIVE] = (PE_after[ACTIVE]-PE_before[ACTIVE]<energy)
            
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
                    CNVG[FINAL] = abs(PE_after[FINAL]-PE_before[FINAL]-energy)<=(energy*CNVG_T)
                    cf = abs(PE_after[FINAL]-PE_before[FINAL]-energy)<=(energy*CNVG_T)
                    
                    MLD[CNVG] = dz_Mixed[CNVG]
                    FINISHED[CNVG] = True
                    FINAL[CNVG] = False
                    
                    TOO_HIGH = (PE_after[FINAL]-PE_before[FINAL])>energy
                    TOO_LOW  =  (PE_after[FINAL]-PE_before[FINAL])<energy
                    DZup = DZup[~cf]
                    DZlo = DZlo[~cf]
                    DZup[TOO_HIGH]=DZ[-1,~cf][TOO_HIGH]
                    DZlo[TOO_LOW]=DZ[-1,~cf][TOO_LOW]
                    

                else:
                    print(IT,np.sum(FINAL),' #not converged')
                    print(PE_after[FINAL],PE_before[FINAL],PE_after[FINAL]-PE_before[FINAL],energy)
                    print(DZup,DZlo,DZ[-1,:])
                    print(Z_U[:,FINAL])
                    MLD[FINAL] = dz_Mixed[FINAL]
                    FINAL[FINAL] = False
                    asdf

        self.mld = MLD

