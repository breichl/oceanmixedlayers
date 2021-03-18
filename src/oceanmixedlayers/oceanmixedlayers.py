from .threshold import threshold as _threshold
from .gradient import gradient as _gradient
from .holtetalley import holtetalley as _holtetalley
from .energy import energy as _energy
from .energy2mix import energy2mix as _energy2mix

class oceanmixedlayers():
    """
    Main class for ocean mixed layers computations.
    """

    def __init__(self):
        self.MLD = 0.0
        
    def threshold(coordinate, value, delta=0.0, ref=0.0):

        mld = _threshold.threshold_mld_fixedcoord(coordinate, value, delta, ref)

        return mld

    def gradient(coordinate, value, critical_gradient=1.e-5):

        mld = _gradient.gradient_mld_fixedcoord(coordinate, value, critical_gradient)

        return mld

    def linearfit(coordinate, value, error_tolerance=1.e-10):

        mld = _gradient.linearfit_mld_fixedcoord(coordinate, value, error_tolerance)

        return mld

    def holtetalley(pressure, salinity, temperature, density):

        mld = _holtetalley.algorithm_mld(pressure, salinity, temperature, density)

        return mld

    def pe_anomaly_density(z_c, thck, ptntl_rho_layer, ptntl_rho_grad=0.0, energy=25., gradient=False):
        if (not gradient):
            ptntl_rho_grad = ptntl_rho_layer*0.
        else:
            if len(ptntl_rho_grad.shape)==0:
                print('Need to pass ptntl_rho_grad to pe_anomaly_density if gradient=True')
                asdf
        mld = _energy.energetic_mixing_depth_Rho0_Linear_nd(ptntl_rho_layer,ptntl_rho_grad,z_c, thck, energy)
        return mld

    def energy2mix(z_c, thck, ptntl_rho_layer, ptntl_rho_grad=0.0, depth=0., gradient=False):
        if max(depth)>0.:
            print('insert a negative value for depth')
            asdf
        if (not gradient):
            ptntl_rho_grad = ptntl_rho_layer*0.
        else:
            if len(ptntl_rho_grad.shape)==0:
                print('Need to pass ptntl_rho_grad to pe_anomaly_density if gradient=True')
                asdf
        mld = _energy2mix.compute_energy_to_mix(ptntl_rho_layer,ptntl_rho_grad,z_c, thck, depth)
        return mld
