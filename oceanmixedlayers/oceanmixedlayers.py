from .threshold import threshold as _threshold
from .gradient import gradient as _gradient
from .holtetalley import holtetalley as _holtetalley
from .energy import energy as _energy
from .mld_pe_anomaly import mld_pe_anomaly as _mld_pe_anomaly


class oceanmixedlayers:
    """
    Main class for ocean mixed layers computations.
    """

    def __init__(self):
        self.MLD = 0.0

    def threshold(coordinate, value, delta=0.0, ref=0.0):
        """
        Interface to the threshold method mixed layer depth computation.
        Parameters
        ----------
        coordinate: the vertical coordinate 
                    (e.g., depth or pressure)
        value: the value being checked for the threshold mld 
               (e.g., temperature, density, etc.)
        delta: the departure threshold used to find the mld 
               (same units as value)
        ref: the depth where the value reference is set 
             (same units as coordinate)
        
        Returns
        -------
        mld: The mixed layer depth in units of the input coordinate
        """

        mld = _threshold.threshold_mld_fixedcoord(coordinate, value, delta, ref)

        return mld

    def gradient(coordinate, value, critical_gradient=1.0e-5):
        """
        Interface to the gradient method mixed layer depth computation.
        Parameters
        ----------
        coordinate: the vertical coordinate 
                    (e.g., depth or pressure)
        value: the value being checked for the threshold mld 
               (e.g., temperature, density, etc.)
        critical_gradient: the critical gradient for the mld
                           (units as value/coordinate)
        
        Returns
        -------
        mld: The mixed layer depth in units of the input coordinate
        """

        mld = _gradient.gradient_mld_fixedcoord(coordinate, value, critical_gradient)

        return mld

    def linearfit(coordinate, value, error_tolerance=1.0e-10):
        """
        Interface to the linear fit method mixed layer depth computation.
        Parameters
        ----------
        coordinate: the vertical coordinate 
                    (e.g., depth or pressure)
        value: the value being checked for the threshold mld 
               (e.g., temperature, density, etc.)
        error_tolerance: the error from a linear fit used to set the mixed layer slope
                         Solved as summation (value-value_fit)^2
                         (units of value^2)
        
        Returns
        -------
        mld: The mixed layer depth in units of the input coordinate
        """

        mld = _gradient.linearfit_mld_fixedcoord(coordinate, value, error_tolerance)

        return mld

    def holtetalley(pressure, salinity, temperature, density):
        """
        Interface to the Holte and Talley algorithm mixed layer depth computation.

        Parameters
        ----------
        pressure: The pressure (units of Pa)
        salinity: The salinity (units of g/kg)
        temperature: The conservative temperature (units of deg C)
        density: The potential density (kg/m3)

        Returns
        -------
        mld: The mixed layer depth in units of pressure
        """

        mld = _holtetalley.algorithm_mld(pressure, salinity, temperature, density)

        return mld

    def pe_anomaly_density(
        z_c, thck, ptntl_rho_layer, ptntl_rho_grad=0.0, energy=25.0, gradient=False
    ):
        """
        Interface to compute the mld from the PE anomaly based on potential density
        
        Parameters
        ----------
        z_c: The vertical distance from the interface (m)
        thck: The thickness of the layer where density is defined (m)
        ptntl_rho_layer: The mean potential density over the layer (kg/m3)
        ptntl_rho_grad: The gradient of potential density over the layer (kg/m3/m)
        energy: The energy threshold for setting the depth based on PE anomaly (J/m2)
        gradient: a logical to determine if the gradient is used (default is False)
        
        Returns
        -------
        mld: The depth where the value of the PE anomaly equals the defined energy (m)
        """

        if not gradient:
            ptntl_rho_grad = ptntl_rho_layer * 0.0
        else:
            if len(ptntl_rho_grad.shape) == 0:
                print(
                    "Need to pass ptntl_rho_grad to pe_anomaly_density if gradient=True"
                )
                asdf
        mld = _energy.energetic_mixing_depth_Rho0_Linear_nd(
            ptntl_rho_layer, ptntl_rho_grad, z_c, thck, energy
        )
        return mld

    def mld_pe_anomaly(
        z_c, thck, ptntl_rho_layer, ptntl_rho_grad=0.0, depth=0.0, gradient=False
    ):
        """
        Interface to compute the PE anomaly based on potential density from a given depth
        
        Parameters
        ----------
        z_c: The vertical distance from the interface (m)
        thck: The thickness of the layer where density is defined (m)
        ptntl_rho_layer: The mean potential density over the layer (kg/m3)
        ptntl_rho_grad: The gradient of potential density over the layer (kg/m3/m)
        depth: The depth to compute the PE anomalys for (m)
        gradient: a logical to determine if the gradient is used (default is False)
        
        Returns
        -------
        energy: The PE anomaly at the given depth (J/m2)
        """
        if max(depth) > 0.0:
            print("insert a negative value for depth")
            asdf
        if not gradient:
            ptntl_rho_grad = ptntl_rho_layer * 0.0
        else:
            if len(ptntl_rho_grad.shape) == 0:
                print(
                    "Need to pass ptntl_rho_grad to pe_anomaly_density if gradient=True"
                )
                asdf
        mld = _mld_pe_anomaly.compute_energy_to_mix(
            ptntl_rho_layer, ptntl_rho_grad, z_c, thck, depth
        )
        return mld
