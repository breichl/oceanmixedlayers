import numpy as np
import warnings

class threshold():
    '''
    Solve for the coordinate value at the location where the value delta exceeds the prescribed delta.
    '''

    def __init__(self):
        pass

    def threshold_mld_fixedcoord(coordinate, value, delta=0.0, reference=0.0, mask_val=np.NaN, interp=True, interpsurf=True,absdiff=False):
        """Computes the threshold mld with a fixed vertical coordinate"""

        if (coordinate.shape != value.shape):
            RuntimeError("The vertical coordinate must be index 0 of the value array")
            
        num_coord = value.shape[0]
        shape_profs = value.shape[1:]

        if len(shape_profs)==0:
            value = np.atleast_2d(value).T
            coordinate = np.atleast_2d(coordinate).T
            shape_profs = value.shape[1:]

        mask = update_mask(value[0,...],mask_val)

        #Activate any points that aren't initially masked points
        active_mask = (~mask)
        
        #Initialize a mask to indicate profiles that exceed the treshold
        exceeds_mask = np.zeros(shape_profs,dtype=bool)


        # Find the value at the reference coordinate
        value_ref = np.zeros(shape_profs)
        i_c = 0
        while ( (np.sum(active_mask)>0) and (i_c<num_coord-1) ):
            i_c += 1
            
            # Update to search for masked points at level zi
            mask[active_mask] = update_mask(value[i_c,active_mask],mask_val)
            active_mask[mask] = False
            

            exceeds_mask[active_mask] = coordinate[i_c,active_mask]>reference

            if (interpsurf and i_c>1):
                crd_up = coordinate[i_c-1,exceeds_mask]
                crd_dn = coordinate[i_c,exceeds_mask]
                val_up = value[i_c-1,exceeds_mask]
                val_dn = value[i_c,exceeds_mask]
                
                dz = reference - crd_up
                dv_dz = (val_dn-val_up)/(crd_dn-crd_up)
                value_ref[exceeds_mask] = val_up + dz*dv_dz
            else:
                value_ref[exceeds_mask] = value[i_c-1,exceeds_mask]

            active_mask[exceeds_mask] = False
            exceeds_mask[:] = False

            value_ref[active_mask] = value[i_c,active_mask]

        value_threshold = value_ref + delta
        
        #Reset the masks
        
        mask = update_mask(value[0,...],mask_val)
        
        #Activate any points that aren't initially masked points
        active_mask = ~mask
        
        #Initialize a mask to indicate profiles that exceed the treshold
        exceeds_mask = np.zeros(shape_profs,dtype=bool)

        mld = np.zeros(shape_profs)
        mld[mask] = mask_val
        mldi = np.zeros(shape_profs,dtype=int)
        mldi[mask] = -1

        # Find the value at the reference coordinate
        i_c = 0
        while ( (np.sum(active_mask)>0) and (i_c<num_coord-1) ):
            i_c += 1

            # Update to search for masked points at level zi
            mask[active_mask] = update_mask(value[i_c,active_mask],mask_val)
            active_mask[mask] = False
            if absdiff:
                exceeds_mask[active_mask] = abs(value[i_c,active_mask]-value_ref[active_mask])>delta
            else:
                # First check is for value exceeding value at reference depth,
                #  added second check to be sure that we are deeper than the reference value.
                exceeds_mask[active_mask] = (   (value[i_c,active_mask]>value_threshold[active_mask])
                                              & (coordinate[i_c,active_mask]>reference) )
            if interp:
                crd_up = coordinate[i_c-1,exceeds_mask]
                crd_dn = coordinate[i_c,exceeds_mask]
                val_up = value[i_c-1,exceeds_mask]
                val_dn = value[i_c,exceeds_mask]
                if absdiff:
                    dv = abs(value_threshold[exceeds_mask] - val_up)
                    dz_dv = (crd_dn-crd_up)/abs(val_dn-val_up)
                else:
                    dv = value_threshold[exceeds_mask] - val_up
                    dz_dv = (crd_dn-crd_up)/(val_dn-val_up)
                mld[exceeds_mask] = crd_up + dv*dz_dv
                mldi[exceeds_mask] = i_c
            else:
                mld[exceeds_mask] = coordinate[i_c,exceeds_mask]
                mldi[exceeds_mask] = i_c

            active_mask[exceeds_mask] = False
            exceeds_mask[:] = False

            mld[active_mask] = coordinate[i_c,active_mask]
            mldi[active_mask] = i_c

        return mld,mldi

def update_mask(val,mask_val):
    """Updates the mask"""
    return ( np.isnan(val) | (val==mask_val) )
        
