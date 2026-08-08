# Packages
import numpy as np

# Recharge module
from recharge.recharge_pde import RechargeSWME1D

class RechargeSWME1D_CustomIC(RechargeSWME1D):
    """
    RechargeSWME1D variant with recharge-specific custom initial conditions.

    This keeps the base pde.py file untouched and only overrides the 
    initialization hook used by the solver.
    """

    def get_initial_values(
            self,
            order : int,
            initial_condition : str,
            position : float
    ) -> np.ndarray:
        if initial_condition == "smooth_nested_profile_pulse_aggressive":
            return self._smooth_nested_profile_pulse_aggressive(order, position)
        if initial_condition == "smooth_nested_profile_pulse_mild":
            return self._smooth_nested_profile_pulse_mild(order, position)
        if initial_condition == "horton_moment_order_pulse":
            return self._horton_moment_order_pulse(order, position)
        
        # Fallback to the original SWME1D initial conditions
        return super().get_initial_values(order, initial_condition, position)
    
    def _smooth_nested_profile_pulse_aggressive(
            self,
            order : int,
            position : float,
    ) -> np.ndarray:
        """
        Smooth custom benchmark for comparing N=0, N=1, and N=2 recharge models.

        Primitive design:
            h(x) = 1.0 + 0.05 * pulse
            u_m(x) = 1.0 + 0.15 * pulse
            a1(x) = 0.10 * pulse, if order >= 1
            a2(x) = -0.05 * pulse, if order >= 2

        Returned in conserved variabled:
            [h, h * u_m, h * a1, h * a2]
        """
        initial_values = np.zeros(
            self.compute_number_of_variables(order),
            dtype = np.float64,
        )

        # Assumes the current config domain x in [0, 1]
        x0 = 0.5
        sigma = 0.08

        # A simple Gaussian pulse
        pulse = np.exp(-((position - x0) / sigma) ** 2)

        # Primitive variables
        h = 1.0 + 0.05 * pulse
        u_m = 1.0 + 0.15 * pulse

        # Set initial values
        initial_values[0] = h
        initial_values[1] = h * u_m

        if order > 0:
            a1 = 0.10 * pulse
            initial_values[2] = h * a1
        if order > 1:
            a2 = -0.05 * pulse
            initial_values[3] = h * a2

        return initial_values
    
    def _smooth_nested_profile_pulse_mild(
            self,
            order : int,
            position : float,
    ) -> np.ndarray:
        """
        Smooth custom benchmark for comparing N=0, N=1, and N=2 recharge models.

        Primitive design:
            h(x) = 1.0 + 0.05 * pulse
            u_m(x) = 0.8 + 0.15 * pulse
            a1(x) = 0.04 * pulse, if order >= 1
            a2(x) = -0.02 * pulse, if order >= 2

        Returned in conserved variabled:
            [h, h * u_m, h * a1, h * a2]
        """
        initial_values = np.zeros(
            self.compute_number_of_variables(order),
            dtype = np.float64,
        )

        # Assumes the current config domain x in [0, 1]
        x0 = 0.5
        sigma = 0.08

        # A simple Gaussian pulse
        pulse = np.exp(-((position - x0) / sigma) ** 2)

        # Primitive variables
        h = 1.0 + 0.03 * pulse
        u_m = 0.8 + 0.08 * pulse

        # Set initial values
        initial_values[0] = h
        initial_values[1] = h * u_m

        if order > 0:
            a1 = 0.04 * pulse
            initial_values[2] = h * a1
        if order > 1:
            a2 = -0.02 * pulse
            initial_values[3] = h * a2

        return initial_values
    
    
    def _horton_moment_order_pulse(
            self,
            order : int,
            position : float,
    ) -> np.ndarray:
        """
        Smooth initial condition for comparing N=0, N=1, and N=2 recharge models
        under the same Horton rainfall-infiltration forcing.

        This initial condition is intended for the numerical test in which the
        recharge model is run with moment orders N=0, N=1 and N=2. The purpose
        of the test is to check whether the rainfall-infiltration source terms
        behave consistently across the moment hierarchy, while also allowing
        the higher-order models to evolve a non-trivial vertically resolved
        velocity profile.

        The construction is nested across moment order:
            N=0 : h(x) and u_m(x) are initialized.
            N=1 : the same h(x) and u_m(x) are used, and a non-zero first moment
            alpha_1(x) is added.
            N=2 : the same h(x), u_m(x) are used, and a non-zero second moment
            alpha_2(x) is added.

        Primitive variable design:
            h(x) = 1.0 + 0.05 * pulse(x)
            u_m(x) = 0.5 + 0.10 * pulse(x)
            alpha_1(x) = 0.10 * pulse(x), if order >= 1
            alpha_2(x) = -0.05 * pulse(x), if order >= 2

        where pulse(x) is a smooth Gaussian perturbation centered at x = 0.5.
        The defauls parameter values assume the domain x in [0, 1].
        """
        if order not in (0, 1, 2):
            raise NotImplementedError(
                "horton_moment_order_pulse is intended for N=0,1,2."
            )
        
        initial_values = np.zeros(
            self.compute_number_of_variables(order),
            dtype = np.float64,
        )

        # Assumes the test uses x in [0, 1]
        x0 = 0.5
        sigma = 0.08

        # Smooth localized perturbation
        pulse = np.exp(-((position - x0) / sigma) ** 2)

        # Primitive variables
        h = 1.0 + 0.05 * pulse
        u_m = 0.5 + 0.10 * pulse

        # Conserved height and mean-momentum variables.
        initial_values[0] = h
        initial_values[1] = h * u_m

        if order >= 1:
            alpha_1 = 0.10 * pulse
            initial_values[2] = h * alpha_1
        if order >= 2:
            alpha_2 = -0.05 * pulse
            initial_values[3] = h * alpha_2

        return initial_values