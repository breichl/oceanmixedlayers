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

    def energetic_mixing_depth_Linear(R_layer,dRdz,Zc,dZ,Energy,CNVG_T=1.e-5,Debug=False):

        R_layer = np.array(R_layer,dtype=float,ndmin=2)
        dRdz    = np.array(dRdz,dtype=float,ndmin=2)
        Zc      = np.array(Zc,dtype=float,ndmin=2)
        dZ      = np.array(dZ,dtype=float,ndmin=2)
        
        NZ = R_layer.shape[-1]
        Z_U = Zc+dZ/2.
        Z_L = Zc-dZ/2.
        
        IF_ACTIVE = np.ones(np.shape(R_layer)[:-1],dtype='bool')
        IF_ACTIVE[np.isnan(R_layer[...,0])] = False
        
        PE_before  = (
            np.nansum( PE_Kernel_linear_new(R_layer,dRdz,Z_U,Z_L), axis=-1)
        )
        PE_after_target = PE_before + Energy
        
        MLD                 = np.add(np.NaN,np.array(np.zeros(np.shape(R_layer)[:-1])))
        MLD[IF_ACTIVE]      = 0.0
        PE_after            = np.add(np.NaN,np.array(np.zeros(np.shape(R_layer)[:-1])))
        PE_after[IF_ACTIVE] = 0.0
        IT_total            = np.add(np.NaN,np.array(np.zeros(np.shape(R_layer)[:-1])))
        
        for z in range(0,NZ):
            IF_ACTIVE[np.isnan(R_layer[...,z])] = False
            
            #This averages density for the contribution of mixing this layer with others
            R_mixed = (
                np.nansum( R_layer[IF_ACTIVE,:z+1]*dZ[IF_ACTIVE,:z+1], axis=-1) 
                / np.nansum(dZ[IF_ACTIVE,:z+1],axis=-1) 
            )
        
            PE_after[IF_ACTIVE] = (
                PE_Kernel_linear_new(R_mixed,0.0,0.0,
                                     -np.sum(dZ[IF_ACTIVE,:z+1],axis=-1))+
                np.nansum( PE_Kernel_linear_new(R_layer[IF_ACTIVE,z+1:],
                                                dRdz[IF_ACTIVE,z+1:],
                                                Z_U[IF_ACTIVE,z+1:],
                                                Z_L[IF_ACTIVE,z+1:]),
                           axis=-1)
            )
            
            #This is an indexed if loop checking if we have enough energy to mix
            IF_NEXT = (PE_after-PE_before<=Energy) & (IF_ACTIVE)
            MLD[IF_NEXT]+=dZ[IF_NEXT,z]
            
            #This is an indexed if loop if we ran out of energy to mix at this level
            #Note this applies to arrays already shortened to IF_ACTIVE
            IF_LAST = (PE_after[IF_ACTIVE]-PE_before[IF_ACTIVE]>Energy)
            IF_LAST_dim = (PE_after-PE_before>Energy)&(IF_ACTIVE)
            
            IT_track = np.zeros(np.sum(IF_LAST_dim))
            
            if Debug: print('Z, PE Delta: ',z,(PE_before+Energy-PE_after))
            
            if (np.sum(IF_LAST)>=1):
                
                if Debug: print('dZ,Rho,dRdz',dZ[IF_LAST_dim,z],R_layer[IF_LAST_dim,z],dRdz[IF_LAST_dim,z])
                
                iDZm = dZ[IF_LAST_dim,z]*.5
                iZcm = Zc[IF_LAST_dim,z]+dZ[IF_LAST_dim,z]/2-iDZm/2
                iDZn = dZ[IF_LAST_dim,z] - iDZm
                iZcn = Zc[IF_LAST_dim,z]+dZ[IF_LAST_dim,z]/2-iDZm-iDZn/2
                iZcn_U = iZcn+iDZn/2
                iZcn_L = iZcn-iDZn/2
                
                iRm_layer = R_layer[IF_LAST_dim,z]+dRdz[IF_LAST_dim,z]*(iZcm-Zc[IF_LAST_dim,z])
                iRn_layer = R_layer[IF_LAST_dim,z]+dRdz[IF_LAST_dim,z]*(iZcn-Zc[IF_LAST_dim,z])
                dRdz_n = dRdz[IF_LAST_dim,z]
                
                PE_after_it = PE_after[IF_LAST_dim]
                PE_after_target_it = PE_after_target[IF_LAST_dim]
                
                pre_iDZ = dZ[IF_LAST_dim,z]
                if np.ndim(pre_iDZ)==0:
                    pre_iDZ = np.array([pre_iDZ])
                    iDZm = np.array([iDZm])
                    iDZn = np.array([iDZn])
                    iZcm = np.array([iZcm])
                    iZcn = np.array([iZcn])
                    iRm_layer = np.array([iRm_layer])
                    iRn_layer = np.array([iRn_layer])
                    dRdz_n = np.array([dRdz_n])
                    PE_after_it = np.array([PE_after_it])
                    PE_after_target_it = np.array([PE_after_target_it])
                    pre_PE_after = np.copy(PE_after_it)
                    IF_ITERATING = np.ones(np.sum(IF_LAST_dim),dtype='bool')
                    ITERATIONS = np.zeros(np.sum(IF_LAST_dim))
                    
                dZd = np.copy(dZ[IF_LAST_dim,:])
                Zcd = np.copy(Zc[IF_LAST_dim,:])
                Z_Ud = np.copy(Z_U[IF_LAST_dim,:])
                Z_Ld = np.copy(Z_L[IF_LAST_dim,:])
                R_layer_it_U = R_layer[IF_LAST_dim,:z]
                R_layer_it_U = np.atleast_2d(R_layer_it_U)
                R_layer_it_L = R_layer[IF_LAST_dim,z+1:]
                R_layer_it_L = np.atleast_2d(R_layer_it_L)
                
                dRdz_it_U = dRdz[IF_LAST_dim,:z]
                dRdz_it_U = np.atleast_2d(dRdz_it_U)
                dRdz_it_L = dRdz[IF_LAST_dim,z+1:]
                dRdz_it_L = np.atleast_2d(dRdz_it_L)
                
                for IT in range(20):
                    if (np.sum(IF_ITERATING)>=1):
                        
                        IT_track[IF_ITERATING]+=1
                        
                        R_mixed = ( ( np.nansum(R_layer_it_U[IF_ITERATING,:]*dZd[IF_ITERATING,:z],axis=-1)
                                      +iRm_layer[IF_ITERATING]*iDZm[IF_ITERATING])/
                                    (np.sum(dZd[IF_ITERATING,:z],axis=-1)+iDZm[IF_ITERATING]) )
                        PE_after_it[IF_ITERATING] = ( 
                            PE_Kernel_linear_new(R_mixed,0.0,0.0,
                                                 -1.0*np.nansum(dZd[IF_ITERATING,:z], axis=-1)-iDZm[IF_ITERATING])
                            +PE_Kernel_linear_new(iRn_layer[IF_ITERATING],dRdz_n[IF_ITERATING],
                                                  iZcn_U[IF_ITERATING],iZcn_L[IF_ITERATING])
                            +np.nansum( PE_Kernel_linear_new(R_layer_it_L[IF_ITERATING,:],
                                                             dRdz_it_L[IF_ITERATING,:],
                                                             Z_Ud[IF_ITERATING,z+1:],
                                                             Z_Ld[IF_ITERATING,z+1:]), axis=-1 ) 
                        )
                        
                    if Debug: print(IT,iDZm,PE_after_it-PE_after_target_it)
                    IF_ITERATING = abs(PE_after_it-PE_after_target_it)>(Energy*CNVG_T)
                    if (np.sum(IF_ITERATING)>=1):
                        #iDZi =  ((PE_after_target_it[IF_ITERATING] - PE_after_it[IF_ITERATING]) /
                        #         ((PE_after_it[IF_ITERATING] - pre_PE_after[IF_ITERATING])
                        #          /(iDZm[IF_ITERATING] - pre_iDZ[IF_ITERATING])) )
                        A = PE_after_target_it[IF_ITERATING] - PE_after_it[IF_ITERATING]
                        B = PE_after_it[IF_ITERATING] - pre_PE_after[IF_ITERATING]
                        C = iDZm[IF_ITERATING] - pre_iDZ[IF_ITERATING]
                        
                        
                        #if Debug: print(1,iRn_layer,dRdz_n,iZcn_U,iZcn,iZcn_L)
                        #if Debug: print(PE_Kernel_linear(iRn_layer[IF_ITERATING],dRdz_n[IF_ITERATING],
                        #                                                    iZcn_U[IF_ITERATING],iZcn[IF_ITERATING],
                        #                                                    iZcn_L[IF_ITERATING]))
                        #if Debug: print(2,R_layer[:,z],dRdz[:,z],Z_U[z],Zc[z],Z_L[z])
                        #if Debug: print(PE_Kernel_linear(R_layer[IF_ACTIVE,z],
                        #                                              dRdz[IF_ACTIVE,z],Z_U[z],
                        #                                              Zc[z],Z_L[z]))
                        iDZi = A*C/B
                        pre_iDZ[IF_ITERATING] = iDZm[IF_ITERATING]
                        pre_PE_after = np.copy(PE_after_it)
                        iDZm[IF_ITERATING] = np.minimum(np.maximum(0.0,
                                                                   iDZm[IF_ITERATING] + iDZi),
                                                        dZd[IF_ITERATING,z])
                        
                        if (np.max(iDZm>dZd[:,z])):
                            IN = np.where(iDZm>dZd[:,z])[0][0]
                            print('No! Now you have to troubleshoot...')
                            print(iDZi[IN])
                            print(iDZm[IN])
                            print(dZ[z])
                            plt.plot(R_layer_it_U[IF_ITERATING,:][IN],Zc[:z],'k-')
                            plt.plot(iRn_layer[IF_ITERATING][IN],iZcn[IF_ITERATING][IN],'rx')
                            plt.plot(R_layer_it_L[IF_ITERATING,:][IN],Zc[z+1:],'b-')
                            plt.plot(iRn_layer[IF_ITERATING][IN],iZcn[IF_ITERATING][IN],'rx')
                            plt.plot(R_layer_it_L[IF_ITERATING,:][IN],Zc[z+1:],'b-')
                            plt.ylim(-100,0)
                            plt.xlim(1026.2,1026.4)
                            asdf
                            
                        #iZcm[IF_ITERATING] = Zc[z]+dZ[z]/2-iDZm[IF_ITERATING]/2
                        #iDZn[IF_ITERATING] = dZ[z] - iDZm[IF_ITERATING]
                        #iZcn[IF_ITERATING] = Zc[z]+dZ[z]/2-iDZm[IF_ITERATING]-iDZn[IF_ITERATING]/2
                        #iZcn_U[IF_ITERATING] = iZcn[IF_ITERATING]+iDZn[IF_ITERATING]/2
                        #iZcn_L[IF_ITERATING] = iZcn[IF_ITERATING]-iDZn[IF_ITERATING]/2
                        iZcm[IF_ITERATING] = Z_Ud[IF_ITERATING,z]-iDZm[IF_ITERATING]/2.
                        iDZn[IF_ITERATING] = dZd[IF_ITERATING,z] - iDZm[IF_ITERATING]
                        iZcn[IF_ITERATING] = Z_Ud[IF_ITERATING,z]-iDZm[IF_ITERATING]-iDZn[IF_ITERATING]/2.
                        iZcn_U[IF_ITERATING] = Z_Ud[IF_ITERATING,z]-iDZm[IF_ITERATING]
                        iZcn_L[IF_ITERATING] = Z_Ld[IF_ITERATING,z]
                        iRm_layer = R_layer[IF_LAST_dim,z]+dRdz[IF_LAST_dim,z]*(iZcm-Zc[IF_LAST_dim,z])
                        iRn_layer = R_layer[IF_LAST_dim,z]+dRdz[IF_LAST_dim,z]*(iZcn-Zc[IF_LAST_dim,z])
                        ITERATIONS[IF_ITERATING]+=1
                        
                MLD[IF_LAST_dim]+=iDZm
                PE_after[IF_LAST_dim]=PE_after_it
                IT_total[IF_LAST_dim]=IT_track
                IF_ACTIVE[IF_LAST_dim] = False
            
        return MLD,(PE_before+Energy-PE_after),IT_total
                

    def energetic_mixing_depth(R_layer,Zc,dZ,Energy,Debug=False):

        R_layer = np.array(R_layer,dtype=float,ndmin=2)
        Zc      = np.array(Zc,dtype=float,ndmin=2)
        dZ      = np.array(dZ,dtype=float,ndmin=2)
        
        NZ = R_layer.shape[-1]
        Z_U = Zc+dZ/2.
        Z_L = Zc-dZ/2.
        
        IF_ACTIVE = np.ones(np.shape(R_layer)[:-1],dtype='bool')
        IF_ACTIVE[np.isnan(R_layer[...,0])] = False
        
        
        PE_before  = np.nansum( PE_Kernel(R_layer,Z_U,Z_L),
                                axis=-1)
        PE_after_target = PE_before+Energy
        
        MLD            = np.add(np.NaN,np.array(np.zeros(np.shape(R_layer)[:-1])))
        MLD[IF_ACTIVE] = 0.0
        PE_after            = np.add(np.NaN,np.array(np.zeros(np.shape(R_layer)[:-1])))
        PE_after[IF_ACTIVE] = 0.0
        
        IT_total  = np.add(np.NaN,np.array(np.zeros(np.shape(R_layer)[:-1])))
        
        for z in range(0,NZ):
            IF_ACTIVE[np.isnan(R_layer[...,z])] = False
            if (np.sum(IF_ACTIVE)>0):
                #This averages density for the contribution of mixing this layer with others
                R_mixed = (np.nansum(R_layer[IF_ACTIVE,:z+1]*dZ[IF_ACTIVE,:z+1],axis=-1) /
                           np.nansum(dZ[IF_ACTIVE,:z+1],axis=-1) )
                
                PE_after[IF_ACTIVE] =        (np.nansum( PE_Kernel(R_mixed,
                                                                   Z_U[IF_ACTIVE,:z+1],
                                                                   Z_L[IF_ACTIVE,:z+1]),
                                                         axis=-1) +
                                              np.nansum( PE_Kernel(R_layer[IF_ACTIVE,z+1:],
                                                                   Z_U[IF_ACTIVE,z+1:],
                                                                   Z_L[IF_ACTIVE,z+1:]),
                                                         axis=-1) )
                
                #This is an indexed if loop checking if we have enough energy to mix
                IF_NEXT = (PE_after-PE_before<Energy) & (IF_ACTIVE)
                MLD[IF_NEXT]+=dZ[IF_NEXT,z]
                
                #This is an indexed if loop if we ran out of energy to mix at this level
                #Note this applies to arrays already shortened to IF_ACTIVE
                IF_LAST = (PE_after[IF_ACTIVE]-PE_before[IF_ACTIVE]>=Energy)
                IF_LAST_dim = (PE_after-PE_before>Energy)&(IF_ACTIVE)
                
                IT_track = np.zeros(np.sum(IF_LAST_dim))
                
                if Debug: print(z,PE_after-PE_before)
                
                if (np.sum(IF_LAST)>=1):
                    
                    iDZm = dZ[IF_LAST_dim,z]*.5
                    iZcm = Zc[IF_LAST_dim,z]+dZ[IF_LAST_dim,z]/2-iDZm/2
                    iDZn = dZ[IF_LAST_dim,z] - iDZm
                    iZcn = Zc[IF_LAST_dim,z]+dZ[IF_LAST_dim,z]/2-iDZm-iDZn/2
                    iZcn_U = iZcn+iDZn/2
                    iZcn_L = iZcn-iDZn/2
                    
                    iRm_layer = R_layer[IF_LAST_dim,z]
                    iRn_layer = R_layer[IF_LAST_dim,z]
                    
                    PE_after_it = PE_after[IF_LAST_dim]
                    PE_after_target_it = PE_after_target[IF_LAST_dim]
                    
                    pre_iDZ = dZ[IF_LAST_dim,z]
                    if np.ndim(pre_iDZ)==0:
                        pre_iDZ = np.array([pre_iDZ])
                        iDZm = np.array([iDZm])
                        iDZn = np.array([iDZn])
                        iZcm = np.array([iZcm])
                        iZcn = np.array([iZcn])
                        iRm_layer = np.array([iRm_layer])
                        iRn_layer = np.array([iRn_layer])
                        PE_after_it = np.array([PE_after_it])
                        PE_after_target_it = np.array([PE_after_target_it])
                        pre_PE_after = np.copy(PE_after_it)
                        IF_ITERATING = np.ones(np.sum(IF_LAST_dim),dtype='bool')
                        ITERATIONS = np.zeros(np.sum(IF_LAST_dim))
                        
                    dZd = np.copy(dZ[IF_LAST_dim,:])
                    Zcd = np.copy(Zc[IF_LAST_dim,:])
                    Z_Ud = np.copy(Z_U[IF_LAST_dim,:])
                    Z_Ld = np.copy(Z_L[IF_LAST_dim,:])
                    R_layer_it_U = R_layer[IF_LAST_dim,:z]
                    R_layer_it_U = np.atleast_2d(R_layer_it_U)
                    R_layer_it_L = R_layer[IF_LAST_dim,z+1:]
                    R_layer_it_L = np.atleast_2d(R_layer_it_L)
                    
                    for IT in range(20):
                        if (np.sum(IF_ITERATING)>=1):
                            
                            IT_track[IF_ITERATING]+=1
                            
                            R_mixed = ( ( np.nansum(R_layer_it_U[IF_ITERATING,:]*dZd[IF_ITERATING,:z],axis=-1)
                                          +iRm_layer[IF_ITERATING]*iDZm[IF_ITERATING])/
                                        (np.sum(dZd[IF_ITERATING,:z],axis=-1)+iDZm[IF_ITERATING]) )
                            PE_after_it[IF_ITERATING] =      ( np.nansum( PE_Kernel( R_mixed,
                                                                                     Z_Ud[IF_ITERATING,:z],
                                                                                     Z_Ld[IF_ITERATING,:z]),
                                                                          axis=-1)
                                                               +PE_Kernel(R_mixed,
                                                                          Z_Ud[IF_ITERATING,z],
                                                                          iZcn_U[IF_ITERATING])
                                                               +PE_Kernel(iRn_layer[IF_ITERATING],
                                                                          iZcn_U[IF_ITERATING],
                                                                          iZcn_L[IF_ITERATING])
                                                               +np.nansum( PE_Kernel(R_layer_it_L[IF_ITERATING,:],
                                                                                     Z_Ud[IF_ITERATING,z+1:],
                                                                                     Z_Ld[IF_ITERATING,z+1:]),
                                                                           axis=-1 ) )
                            
                        IF_ITERATING = abs(PE_after_it-PE_after_target_it)>(Energy*.00001)
                        if (np.sum(IF_ITERATING)>=1):
                            #iDZi =  ((PE_after_target_it[IF_ITERATING] - PE_after_it[IF_ITERATING]) /
                            #         ((PE_after_it[IF_ITERATING] - pre_PE_after[IF_ITERATING])
                            #          /(iDZm[IF_ITERATING] - pre_iDZ[IF_ITERATING])) )
                            A = PE_after_target_it[IF_ITERATING] - PE_after_it[IF_ITERATING]
                            B = PE_after_it[IF_ITERATING] - pre_PE_after[IF_ITERATING]
                            C = iDZm[IF_ITERATING] - pre_iDZ[IF_ITERATING]
                            
                            iDZi = A*C/B
                            pre_iDZ[IF_ITERATING] = iDZm[IF_ITERATING]
                            pre_PE_after = np.copy(PE_after_it)
                            iDZm[IF_ITERATING] = np.minimum(np.maximum(0.0,
                                                                       iDZm[IF_ITERATING] + iDZi),
                                                            dZd[IF_ITERATING,z])
                            # There are issues (fairly sure precision driven) in the calculation that result
                            # in differences in predicted mixing where the code thinks it has enough energy to mix
                            # a layer in one calculation but then it thinks it does not in another calculation
                            # due to the order the calculations are done in.  I think fixing this will require
                            # quite a bit of re-engineering of this code.  So, for now when this happens we
                            # will stop iterating those points and set MLD.
                            #IF_ITERATING[(abs(iDZi)<1.e-6)&((iDZm==0.0)|(iDZm==dZ[z]))] = False
                            #This issue was because the pressure was float-type 32, instead of 64.
                            # It has been corrected before calling this routine.
                            if (np.max(iDZm>dZd[:,z])):
                                IN = np.where(iDZm>dZd[:,z])[0][0]
                                print('No! Now you have to troubleshoot...')
                                print(iDZi[IN])
                                print(iDZm[IN])
                                print(dZ[z])
                                plt.plot(R_layer_it_U[IF_ITERATING,:][IN],Zc[:z],'k-')
                                plt.plot(iRn_layer[IF_ITERATING][IN],iZcn[IF_ITERATING][IN],'rx')
                                plt.plot(R_layer_it_L[IF_ITERATING,:][IN],Zc[z+1:],'b-')
                                plt.plot(iRn_layer[IF_ITERATING][IN],iZcn[IF_ITERATING][IN],'rx')
                                plt.plot(R_layer_it_L[IF_ITERATING,:][IN],Zc[z+1:],'b-')
                                plt.ylim(-100,0)
                                plt.xlim(1026.2,1026.4)
                                asdf
                                
                            #iZcm[IF_ITERATING] = Zc[z]+dZ[z]/2-iDZm[IF_ITERATING]/2
                            #iDZn[IF_ITERATING] = dZ[z] - iDZm[IF_ITERATING]
                            #iZcn[IF_ITERATING] = Zc[z]+dZ[z]/2-iDZm[IF_ITERATING]-iDZn[IF_ITERATING]/2
                            #iZcn_U[IF_ITERATING] = iZcn[IF_ITERATING]+iDZn[IF_ITERATING]/2
                            #iZcn_L[IF_ITERATING] = iZcn[IF_ITERATING]-iDZn[IF_ITERATING]/2
                            iZcm[IF_ITERATING] = Z_Ud[IF_ITERATING,z]-iDZm[IF_ITERATING]/2.
                            iDZn[IF_ITERATING] = dZd[IF_ITERATING,z] - iDZm[IF_ITERATING]
                            iZcn[IF_ITERATING] = Z_Ud[IF_ITERATING,z]-iDZm[IF_ITERATING]-iDZn[IF_ITERATING]/2.
                            iZcn_U[IF_ITERATING] = Z_Ud[IF_ITERATING,z]-iDZm[IF_ITERATING]
                            iZcn_L[IF_ITERATING] = Z_Ld[IF_ITERATING,z]
                            iRm_layer = R_layer[IF_LAST_dim,z]
                            iRn_layer = R_layer[IF_LAST_dim,z]
                            ITERATIONS[IF_ITERATING]+=1
                            
                    MLD[IF_LAST_dim]+=iDZm
                    PE_after[IF_LAST_dim]=PE_after_it
                    IT_total[IF_LAST_dim]=IT_track
                    IF_ACTIVE[IF_LAST_dim] = False

        return MLD,(PE_before+Energy-PE_after),IT_total

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
