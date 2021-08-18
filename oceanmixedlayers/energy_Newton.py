import numpy as np
import gsw as gsw
import warnings

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
        IT_total         = np.add(np.NaN,np.array(np.zeros(ND)))
        IT_total[ACTIVE] = 0
        
        z=-1
        while(z<NZ-1 and np.sum(ACTIVE)>0):
            z += 1
            
            CNVG = np.zeros(ND,dtype='bool')
            ACTIVE[np.isnan(Rho0_layer[z,...])] = False
            FINISHED[np.isnan(Rho0_layer[z,...])] = True
            FINAL = np.zeros(ND,dtype='bool')
            
            PE_before[ACTIVE]  = (
                np.sum(PE_Kernel_linear(Rho0_layer[:z+1,ACTIVE],
                                            dRho0dz_layer[:z+1,ACTIVE],
                                            Z_U[:z+1,ACTIVE],
                                            Z_L[:z+1,ACTIVE]), axis=0)
            )
            
            Rho0_Mixed[:z+1,ACTIVE],_ = (
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

            # First guess for an iteration using Newton's method
            X = dZ[z,FINAL] * 0.5

            IT = 0
            IT_Lim = 10
            while (np.sum(FINAL)>=1 and IT<=IT_Lim):
                IT+=1

                #Needed for the Newton iteration
                #Within the iteration so that the size updates with each iteration.
                #In principle we might move this outside in favor of more logical indexing
                # but since the iteration converges in 2 steps it is probably OK.
                R1,D1 = MixLayers(Rho0_layer[:,FINAL],dZ[:,FINAL],z)
                R2,D2 = Rho0_layer[z,FINAL],dZ[z,FINAL]
                
                Ca  = -R2
                Cb  = -(R1 * D1 + R2 * (2. * D1))
                D   = D1**2
                Cc  = -(R1 * D1 * (2. * D1) + (R2 * D))
                Cd  = -R1 * (D1 * D)
                Ca2 = R2
                Cb2 = R2 * (2. * D1)
                C   = D2**2 + D1**2 + 2. * (D1 * D2)
                Cc2 = R2 * (D - C)

                # We are trying to solve the function:
                # F(x) = G(x)/H(x)+I(x)
                # for where F(x) = PE+PE_threshold, or equivalently for where
                # F(x) = G(x)/H(x)+I(x) - (PE+PE_threshold) = 0
                # We also need the derivative of this function for the Newton's method iteration
                # F'(x) = (G'(x)H(x)-G(x)H'(x))/H(x)^2 + I'(x)
                # G and its derivative
                Gx = 0.5 * (Ca * (X*X*X) + Cb * X**2 + Cc * X + Cd)
                Gpx = 0.5 * (3. * (Ca * X**2) + 2. * (Cb * X) + Cc)
                # H, its inverse, and its derivative
                Hx = D1 + X
                iHx = 1. / Hx
                Hpx = 1.
                # I and its derivative
                Ix = 0.5 * (Ca2 * X**2 + Cb2 * X + Cc2)
                Ipx = 0.5 * (2. * Ca2 * X + Cb2)
                
                # The Function and its derivative:
                PE_Mixed = Gx * iHx + Ix
                Fgx = PE_Mixed - (PE_before[FINAL] + energy)
                Fpx = (Gpx * Hx - Hpx * Gx) * iHx**2 + Ipx
                    
                # Check if our solution is within the threshold bounds, if not update
                # using Newton's method.  This appears to converge almost always in
                # one step because the function is very close to linear in most applications.
                CNVG_ = (abs(Fgx) < energy * CNVG_T)
                CNVG[FINAL] = CNVG_
                #Disable any that have converged and add to output
                FINAL[CNVG] = False
                MLD[CNVG] += X[CNVG_]

                #Update those that haven't converged
                nCNVG_ = ~CNVG_
                X2 = X[nCNVG_] - Fgx[nCNVG_] / Fpx[nCNVG_]
                X = X2
                
                IT_FAILED = (X2 < 0.)|(X2 > D2[nCNVG_])
                # The iteration seems to be robust, but we need to do something *if*
                # things go wrong... How should we treat failed iteration?
                # Present solution: Fail the entire algorithm.
                if np.sum(IT_FAILED)>0:
                    print(IT,'Iteration failed in energy_newiteration')
                    asdf
                    
                if (IT==IT_Lim and np.sum(FINAL)>0):
                    print(IT,np.sum(FINAL),' #not converged')
                    asdf

        self.mld = MLD

