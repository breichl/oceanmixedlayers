import numpy as np
import warnings

class extrema():
    '''
    Solve for the coordinate value at the location where the value is equal to some extrema 
    (e.g., min/max).
    '''

    def __init__(self):
        pass

    def maxval_mld_fixedcoord(coordinate, value, mask_val=np.NaN):
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

        mld = np.zeros(shape_profs)
        mld[mask] = mask_val
        mldi = np.zeros(shape_profs,dtype=int)
        mldi[mask] = -1

        maxval = np.nanmax(value,axis=0)

        i_c = -1
        while ( (np.sum(active_mask)>0) and (i_c<num_coord-1) ):
            i_c += 1
            
            # Update to search for masked points at level zi
            mask[active_mask] = update_mask(value[i_c,active_mask],mask_val)
            active_mask[mask] = False
            

            exceeds_mask[active_mask] = value[i_c,active_mask]==maxval[active_mask]

            mld[exceeds_mask] = coordinate[i_c,exceeds_mask]
            mldi[exceeds_mask] = i_c
            
            #keep searching in case a deeper point is found
            #active_mask[exceeds_mask] = False
            exceeds_mask[:] = False

        return mld,mldi

def update_mask(val,mask_val):
    """Updates the mask"""
    return ( np.isnan(val) | (val==mask_val) )
        
