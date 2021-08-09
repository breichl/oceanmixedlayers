import numpy as np
import warnings

class gradient():
    '''
    Solve for the coordinate value at the location where the value gradient exceeds the prescribed threshold gradient.
    '''

    def __init__(self):
        pass

        
    def gradient_mld_fixedcoord(coordinate, value, critical_gradient, mask_val=np.NaN,smooth=True):
        """Computes the gradient mld with a fixed vertical coordinate"""

        if (coordinate.shape != value.shape):
            RuntimeError("The vertical coordinate must be index 0 of the value array")

        num_coord = value.shape[0]
        shape_profs = value.shape[1:]

        if len(shape_profs)==0:
            coordinate = np.atleast_2d(coordinate).T
            value = np.atleast_2d(value).T
            shape_profs = value.shape[1:]
            
        dv_dc = -grad(coordinate,value,smooth)
        num_coord = dv_dc.shape[0]
        
        mask = update_mask(value[0,...],mask_val)

        #Activate any points that aren't initially masked points
        active_mask = (~mask)
        
        #Initialize a mask to indicate profiles that exceed the treshold
        exceeds_mask = np.zeros(shape_profs,dtype=bool)

        mld = np.zeros(shape_profs)
        mldi = np.zeros(shape_profs,dtype=int)
        mld[mask] = mask_val
        mldi[mask] = -1
        
        # Find the depth where the threshold gradient lies
        i_c = 0
        ##First check if the top exceeds (possible due to smoothing)
        #exceeds_mask[active_mask] = dv_dc[i_c,active_mask]>critical_gradient
        #mld[exceeds_mask] = coordinate[i_c,exceeds_mask]
        #mldi[exceeds_mask] = i_c
        #active_mask[exceeds_mask] = False
        #exceeds_mask[:] = False
        while ( (np.sum(active_mask)>0) and (i_c<num_coord-2) ):
            #i_c += 1

            exceeds_mask[active_mask] = abs(dv_dc)[i_c,active_mask]>critical_gradient


            # Add interpolate option with this code
            #crd_up = coordinate[i_c-1,exceeds_mask]
            #crd_dn = coordinate[i_c,exceeds_mask]
            #val_up = dv_dc[i_c-1,exceeds_mask]
            #val_dn = dv_dc[i_c,exceeds_mask]
            
            #dv = critical_gradient - val_up
            #dz_dv = (crd_dn-crd_up)/(val_dn-val_up)

            mld[exceeds_mask] = coordinate[i_c+1,exceeds_mask]
            mldi[exceeds_mask] = i_c+1

            active_mask[exceeds_mask] = False
            exceeds_mask[:] = False
            i_c += 1

        # If no points hit the critical gradient, return the depth of maximum gradient
        dv_dc_max = np.nanmax(abs(dv_dc),axis=0)
        i_c=0

        while ( (np.sum(active_mask)>0) and (i_c<num_coord) ):
            exceeds_mask[active_mask] = abs(dv_dc)[i_c,active_mask]==abs(dv_dc_max)[active_mask]

            mld[exceeds_mask] = coordinate[i_c+1,exceeds_mask]
            mldi[exceeds_mask] = i_c+1
            active_mask[exceeds_mask] = False
            exceeds_mask[:] = False
            i_c+=1
        return mld, mldi

    def max_gradient(coordinate, value, critical_gradient, mask_val=np.NaN,smooth=True):
        """Computes the gradient mld with a fixed vertical coordinate"""

        if (coordinate.shape != value.shape):
            RuntimeError("The vertical coordinate must be index 0 of the value array")

        num_coord = value.shape[0]
        shape_profs = value.shape[1:]

        if len(shape_profs)==0:
            coordinate = np.atleast_2d(coordinate).T
            value = np.atleast_2d(value).T
            shape_profs = value.shape[1:]
            
        dv_dc = grad(coordinate,value,smooth)
        num_coord = dv_dc.shape[0]

        mask = update_mask(value[0,...],mask_val)

        #Activate any points that aren't initially masked points
        active_mask = (~mask)
        
        #Initialize a mask to indicate profiles that exceed the treshold
        exceeds_mask = np.zeros(shape_profs,dtype=bool)

        mld = np.zeros(shape_profs)
        mldi = np.zeros(shape_profs,dtype=int)
        mld[mask] = mask_val
        mldi[mask] = -1
        
        dv_dc_max = np.nanmax(dv_dc,axis=0)
        i_c=0

        while ( (np.sum(active_mask)>0) and (i_c<num_coord) ):
            exceeds_mask[active_mask] = (dv_dc)[i_c,active_mask]==(dv_dc_max)[active_mask]

            mld[exceeds_mask] = coordinate[i_c+1,exceeds_mask]
            mldi[exceeds_mask] = i_c+1
            #Keep all points active in case the exceeds mask is hit again
            #active_mask[exceeds_mask] = False
            exceeds_mask[:] = False
            i_c+=1
        return mld, mldi


    def linearfit_mld_fixedcoord(coordinate, value, error_tolerance, mask_val=np.NaN,smooth=True):
        """
        Computes the mld from fiting a linear slope to the thermocline and mixed layer 
        with a fixed vertical coordinate
        """
        if (coordinate.shape != value.shape):
            RuntimeError("The vertical coordinate must be same shape as the value array")

        num_coord = value.shape[0] -1 #Should this be -1?
        shape_profs = value.shape[1:]

        if len(shape_profs)==0:
            value = np.atleast_2d(value).T
            coordinate = np.atleast_2d(coordinate).T
            shape_profs = value.shape[1:]
            
        mask = update_mask(value[0,...],mask_val)
        #Activate any points that aren't initially masked points
        active_mask = (~mask)
        exceeds_mask = np.zeros(shape_profs,dtype=bool)
        
        mld = np.zeros(shape_profs)
        mld[mask] = mask_val
        mldi = np.zeros(shape_profs,dtype=int)
        mldi[mask] = -1

        error = np.zeros(value.shape)
        error[:,mask] = mask_val

        # Compute error of linear fit of data

        i_c = 0
        while ( (np.sum(active_mask)>0) and (i_c<num_coord-1) ):
            i_c += 1

            # Update to search for masked points at level zi
            mask[active_mask] = update_mask(value[i_c,active_mask],mask_val)
            active_mask[mask] = False
            
            slope, intercept = nd_linefit(coordinate[:i_c+1,active_mask],value[:i_c+1,active_mask])
            model = (intercept+slope*coordinate[:i_c+1,active_mask])
            if len(model.shape)==0:
                model = np.atleast_2d(model).T
            error[i_c,active_mask] = np.sum((value[:i_c+1,active_mask]-model)**2,axis=0)

        error = error/np.nansum(error,axis=0)

        mask = update_mask(value[0,...],mask_val)
        #Activate any points that aren't initially masked points
        active_mask = (~mask)
        ml_slope = np.zeros(shape_profs)
        ml_intercept = np.zeros(shape_profs)
        ml_slope[mask] = mask_val
        ml_intercept[mask] = mask_val

        i_c=0#Should this be i_c=-1?
        while ( (np.sum(active_mask)>0) and (i_c<num_coord-1) ):
            i_c+=1

            exceeds_mask[active_mask] = error[i_c,active_mask]>error_tolerance
            slope, intercept = nd_linefit(coordinate[:i_c,exceeds_mask],value[:i_c,exceeds_mask])

            ml_slope[exceeds_mask] = slope
            ml_intercept[exceeds_mask] = intercept
            
            active_mask[exceeds_mask] = False
            exceeds_mask[:] = False
        
        # Fine the ___cline model
        cl_slope = np.zeros(shape_profs)
        cl_intercept = np.zeros(shape_profs)
        cl_slope[mask] = mask_val
        cl_intercept[mask] = mask_val

        dv_dc = -grad(coordinate,value,smooth)
        coordinate_sm = 1./3.*(coordinate[2:,...]+coordinate[1:-1,...]+coordinate[:-2,...])
        dv_dc_max = np.nanmax(abs(dv_dc),axis=0)

        mask = update_mask(dv_dc_max,mask_val)
        #Activate any points that aren't initially masked points
        active_mask = (~mask)
        i_c=-1

        while ( (np.sum(active_mask)>0) and (i_c<num_coord-1) ):
            i_c+=1
            exceeds_mask[active_mask] = abs(dv_dc)[i_c,active_mask]==abs(dv_dc_max)[active_mask]

            #Note +1 because of smoothing above
            slope, intercept = nd_linefit(coordinate[i_c:i_c+3,exceeds_mask],
                                          value[i_c:i_c+3,exceeds_mask])

            cl_slope[exceeds_mask] = slope
            cl_intercept[exceeds_mask] = intercept
            
            active_mask[exceeds_mask] = False
            exceeds_mask[:] = False

        #print(ml_intercept,ml_slope)
        #print(cl_intercept,cl_slope)

        ml_values = ml_intercept+ml_slope*coordinate
        cl_values = cl_intercept+cl_slope*coordinate
        
        if len(ml_values.shape)==0:
            ml_values = np.atleast_2d(ml_values).T
            cl_values = np.atleast_2d(cl_values).T

        
        closest = np.nanmin(abs(ml_values-cl_values),axis=0)
        maximum = np.nanmax(ml_values-cl_values,axis=0)
        minimum = np.nanmin(ml_values-cl_values,axis=0)

        mask = update_mask(value[0,...],mask_val)
        #Activate any points that aren't initially masked points
        active_mask = (~mask)
        i_c=0
        while ( (np.sum(active_mask)>0) and (i_c<num_coord) ):
            exceeds_mask[active_mask] = (closest[active_mask]==abs(ml_values[i_c,active_mask]-
                                                     cl_values[i_c,active_mask]))
            mld[exceeds_mask] = coordinate[i_c,exceeds_mask]
            mldi[exceeds_mask] = i_c
            active_mask[exceeds_mask] = False
            exceeds_mask[:] = False
            i_c+=1
        mld[np.sign(maximum)==np.sign(minimum)] = 0.
        mldi[np.sign(maximum)==np.sign(minimum)] = -2
        
        return mld, mldi
        
def update_mask(val,mask_val):
    """Updates the mask"""
    return ( np.isnan(val) | (val==mask_val) )
        
def grad(coordinate,value,smooth):
    one_thrd = 1./3.
    #dv_dc = np.zeros(np.shape(value))
    # central difference gradient w/ special treatment of edges
    #dv_dc[0,...] = (value[0,...]-value[1,...])/(coordinate[0,...]-coordinate[1,...])
    #dv_dc[1:-1,...] = (value[0:-2,...]-value[2:,...])/(coordinate[0:-2,...]-coordinate[2:,...])
    #dv_dc[-1,...] = (value[-2,...]-value[-1,...])/(coordinate[-2,...]-coordinate[-1,...])
    
    dv_dc = (value[:-1,...]-value[1:,...])/(coordinate[:-1,...]-coordinate[1:,...])

    # Smoothing following the Holte and Taley .m code
    if smooth:
        return one_thrd*(dv_dc[0:-2,...]+dv_dc[1:-1,...]+dv_dc[2:,...])
    else:
        return dv_dc

def nd_linefit(coordinate,value):

    val_mean = np.nanmean(value,axis=0)
    val_std = np.nanstd(value,axis=0)
    coor_mean = np.nanmean(coordinate,axis=0)
    coor_std = np.nanstd(coordinate,axis=0)

    nzero = (coor_std!=0)&(val_std!=0)

    covariance = np.zeros(np.shape(val_mean))
    covariance[nzero] = np.nanmean((coordinate[:,nzero]-coor_mean[nzero])
                                   *(value[:,nzero]-val_mean[nzero]),axis=0)

    correlation = np.zeros(np.shape(val_mean))
    correlation[nzero] = covariance[nzero]/(val_std[nzero]*coor_std[nzero])

    slope = np.zeros(np.shape(val_mean))
    slope[nzero] = covariance[nzero]/(coor_std[nzero]**2)

    intercept = val_mean - coor_mean*slope

    return slope, intercept
