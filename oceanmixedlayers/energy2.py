import numpy as np
import warnings

class energy():
    '''
    Solve for the mld from a PE anomaly consideration
    '''

    def __init__(self):
        pass

    def energy_anomaly_mld_density(z_c,#Central z position relative to sea surface (negative) (m)
                                   thck,#Thickness (positive) (m)
                                   ptntl_rho_layer,#mean potential density of layer (kg/m3)
                                   ptntl_rho_grad = 0.,#gradient of potential density across layer (kg/m3/m)
                                   energy=1.0,#energy threshold (J/m2)
                                   threshold=1.e-5,#convergence threshold (nd)
                                   gradient=False,
                                   mask_val=np.NaN
    ):
        """Computes the threshold mld with a fixed vertical coordinate"""
        num_coord = ptntl_rho_layer.shape[0]
        shape_profs= ptntl_rho_layer.shape[1:]

        # Construct nd array of interfaces (1st index is depth, 2+ indexes can be for 2d/3d arrays)
        z_i=np.zeros([num_coord+1,]+list(shape_profs))
        z_i[:-1,...] = z_c-thck/2
        z_i[-1,...]=z_c[-1,...]+thck[-1,...]/2
        
        ptntl_rho_mixed = np.copy(ptntl_rho_layer)
        if gradient:
            if (ptntl_rho_grad!=ptntl_rho_layer):
                RuntimeError("If gradient is True for energy_anomaly_mld_density, then a ptntl_rho_grad must be the same"+
                             " size as ptntl_rho_layer")
        else:
            ptntl_rho_grad = np.zeros(ptntl_rho_layer.shape)

        active_mask = np.ones(shape_profs,dtype='bool')
        final_mask = np.zeros(shape_profs,dtype='bool')
        
        mask = update_mask(ptntl_rho_layer[0,...],mask_val)
        active_mask = ~mask
        mld = np.add(np.NaN,np.zeros(shape_profs));mld[active_mask]=0.0
        delta_energy = np.add(np.NaN,np.zeros(shape_profs));delta_energy[active_mask]=0.0
        pe_mixed = np.add(np.NaN,np.zeros(shape_profs));pe_mixed[active_mask]=0.0
        pe_unmixed = np.add(np.NaN,np.zeros(shape_profs));pe_unmixed[active_mask]=0.0
        dz_mixed = np.add(np.NaN,np.zeros(shape_profs));dz_mixed[active_mask]=0.0
        iterations = np.add(np.NaN,np.zeros(shape_profs));iterations[active_mask]=0.0

        iz=0
        while(iz<num_coord and np.sum(active_mask)>0):
            
            # Turn off any layers where the density is masked
            mask = update_mask(ptntl_rho_layer[iz,...],mask_val)            
            active_mask[mask] = False
            
            pe_unmixed[active_mask] = np.sum(PE_layer(ptntl_rho_layer[:iz+1,active_mask],
                                                      z_i[:iz+2,active_mask],
                                                      ptntl_rho_grad[:iz+1,active_mask])                                         
                                ,axis=0)
            ptntl_rho_mixed[:iz+1,active_mask] = MixLayers(ptntl_rho_layer[:iz+1,active_mask],thck[:iz+1,active_mask])
            #Note by not passing a gradient we assume no-gradient (because it is mixed)
            pe_mixed[active_mask] = np.sum(PE_layer(ptntl_rho_mixed[:iz+1,active_mask],
                                       z_i[:iz+2,active_mask])
                              ,axis=0)
            final_mask[active_mask] = (pe_mixed[active_mask]-pe_unmixed[active_mask]>=energy)

            if np.sum(final_mask)>0:
                mld[final_mask],delta_energy[final_mask] = FindDepth(ptntl_rho_layer[:iz+1,final_mask],
                                                                     ptntl_rho_grad[:iz+1,final_mask],
                                                                     thck[:iz+1,final_mask],
                                                                     z_i[:iz+2,final_mask],
                                                                     energy)
            active_mask[final_mask] = False
            final_mask[:] = False

            # loop
            iz+=1


        return mld, delta_energy
            


def update_mask(val,mask_val):
    """Updates the mask"""
    return ( np.isnan(val) | (val==mask_val) )

def PE_layer(rho,zi,drhodz=0.0):
    # This is an expression for the integrated PE
    # over a layer assuming a linear change drhodz.
    return  ( rho/2.*(zi[:-1,...]**2-zi[1:,...]**2) + 
              drhodz/3.*(zi[:-1,...]**3-zi[1:,...]**3) - 
              drhodz*(zi[:-1,...]+zi[1:,...])/4.*(zi[:-1,...]**2-zi[1:,...]**2) )

def MixLayers(value,thck):
    # This returns the updated array with homogenized tracers to level nlev
    mixed = np.sum(value*thck,axis=0)/np.sum(thck,axis=0)
    return mixed

def FindDepth(ptntl_rho_layer, ptntl_rho_grad, dz, z_i, energy):
    ti = 0
    if len(ptntl_rho_layer.shape)==0:
        ptntl_rho_layer = np.atleast_2d(ptntl_rho_layer).T
        ptntl_rho_grad = np.atleast_2d(ptntl_rho_grad).T
        dz = np.atleast_2d(dz).T
        z_i = np.atleast_2d(z_i).T
    NZ = ptntl_rho_layer.shape[0]
    NP = ptntl_rho_layer.shape[1:]

    #These are 1d arrays of the size of locations that were exceeded
    dzlo = (np.zeros(NP))
    dzup = (dz[-1,...])
            
    iterator_mask = (np.ones(NP,dtype=bool))
    
    pe_unmixed_above = np.sum( PE_layer(ptntl_rho_layer[:-1,...],z_i[:-1,...],ptntl_rho_grad[:-1,...]),axis=0)
    rho_mixed_above = MixLayers(ptntl_rho_layer[:-1,...],dz[:-1,...])
    dz_mixed_above = np.sum(dz[:-1,...],axis=0)
    zi_mixed = np.zeros([2,]+list(NP))

    while (ti<20):
        ti+=1

        #Guess the bottom intrusion is half of the allowable range
        DZ = 0.5*(dzlo+dzup)
                
        rho_unmixed_bottom = (( ptntl_rho_layer[-1,...] + ptntl_rho_grad[-1,...] * (0.5*dz[-1,...]) ) #value at top        
                             - 0.5 * ptntl_rho_grad[-1,...] * DZ) #adjustment from value at top to value at center                         

        pe_unmixed = pe_unmixed_above+np.sum(PE_layer(rho_unmixed_bottom,z_i[-2:,...],ptntl_rho_grad[-1,...]),axis=0)

        rho_mixed = (rho_mixed_above*dz_mixed_above + rho_unmixed_bottom*DZ)/(dz_mixed_above+DZ)
        zi_mixed[1,...] = dz_mixed_above+DZ
        pe_mixed  = np.sum(PE_layer(rho_mixed,zi_mixed),axis=0)

        TOO_HIGH = (pe_mixed-pe_unmixed)>energy
        TOO_LOW  = (pe_mixed-pe_unmixed)<energy
        dzup[TOO_HIGH]=DZ[TOO_HIGH]
        dzlo[TOO_LOW]=DZ[TOO_LOW]

    return zi_mixed[1,...],pe_mixed-pe_unmixed
