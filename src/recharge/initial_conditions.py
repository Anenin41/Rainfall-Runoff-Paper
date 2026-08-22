# Packages
import numpy as np

# Recharge module
from .recharge_pde import RechargeSWME1D


# Ratio between successive moments in the nested profiles below:
#
#     alpha_{i+1}(x) = MOMENT_DECAY_RATIO * alpha_i(x)
#
# so that alpha_i(x) = alpha_1_amplitude * MOMENT_DECAY_RATIO**(i-1) * pulse(x).
#
# The value is not free, and it is not chosen for smallness. Per
# RESTRUCTURE_PLAN.md Section 6, the N=2 SWME closure loses hyperbolicity on a
# narrow *wedge of moment ratios*, |alpha_2/alpha_1| in [1.14, 1.40], and every
# ray outside that wedge stays hyperbolic at any magnitude - "keep the moments
# small" is the wrong safety criterion, the ratio is what matters. -0.5 is the
# ratio the validated Chapter 5 runs already sit on (they set
# alpha_2 = -0.5*alpha_1), comfortably outside the wedge on the low side.
#
# Extending it geometrically to arbitrary N was checked, not assumed: sweeping
# this ray through the SWME transport matrix (the fragile closure, not HSWME)
# gives max|Im(eigenvalue)| = 0.0 exactly for N = 1..8, out to
# |alpha_1|/sqrt(g*h) = 200 - far past the 0.71 the thesis runs ever reached.
# The one exception is not the model: at the tail of the Gaussian the moments
# vanish, the spectrum becomes degenerate, and LAPACK splits a repeated real
# eigenvalue into a conjugate pair ~1e-17 off the axis. That is fifteen orders
# of magnitude under the suite's own 1e-10 bar, and under a wedge state's ~6e-2.
# The same sweep does reproduce the documented wedge at ratio 1.2, so it is a
# test with teeth rather than one that cannot fail. See
# tests/test_recharge_initial_conditions.py, where all of this is pinned.
#
# The geometric decay also keeps the profile physically sensible as N grows:
# each added moment is a finer vertical detail, and it enters at half the
# amplitude of the one below it, so raising N perturbs the initial velocity
# profile less and less instead of injecting ever more energy at the top of
# the hierarchy.
MOMENT_DECAY_RATIO = -0.5


class RechargeSWME1D_CustomIC(RechargeSWME1D):
    """
    RechargeSWME1D variant with recharge-specific custom initial conditions.

    This keeps the base pde.py file untouched and only overrides the 
    initialization hook used by the solver.

    All three initial conditions below are generic in the moment order N, in
    line with the rest of the suite after the generic coefficient engine landed
    (RESTRUCTURE_PLAN.md Step 3). They are built from one nested construction:
    h(x) and u_m(x) do not depend on N at all, and moment i is seeded at
    `MOMENT_DECAY_RATIO**(i-1)` times the first-moment amplitude. Two properties
    follow, and both are deliberate:

    - **Nesting.** Truncating at any N reproduces the lower-N initial state
      exactly, which is what makes an N-to-N comparison a comparison of the
      *models* rather than of two different problems.
    - **Backward compatibility.** For N = 0, 1, 2 the values are bit-for-bit
      what they were before the generalization (`0.10 * -0.5` is exactly
      `-0.05` in binary floating point, halving being exact), so the validated
      thesis configurations in `swme/config/` are untouched.
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

    def _nested_moment_pulse(
            self,
            order : int,
            position : float,
            *,
            h_ref : float,
            h_amp : float,
            u_ref : float,
            u_amp : float,
            alpha_amp : float,
            x0 : float = 0.5,
            sigma : float = 0.08,
    ) -> np.ndarray:
        """
        Shared generic-N construction behind the three recharge pulses.

        Primitive design, for a Gaussian pulse(x) centered at x0:

            h(x)        = h_ref + h_amp * pulse(x)
            u_m(x)      = u_ref + u_amp * pulse(x)
            alpha_i(x)  = alpha_amp * MOMENT_DECAY_RATIO**(i-1) * pulse(x),
                          for i = 1 .. N

        Returned in the conserved variables the solver expects:

            [h, h*u_m, h*alpha_1, ..., h*alpha_N]

        Parameters
        ----------
        order : int
            moment order N. N = 0 gives plain shallow water (no moment rows);
            any N >= 0 is accepted, matching the generic coefficient engine.
        position : float
            physical position x at which the state is evaluated.
        h_ref, h_amp : float
            background depth and pulse amplitude of the depth perturbation.
        u_ref, u_amp : float
            background mean velocity and pulse amplitude of its perturbation.
        alpha_amp : float
            amplitude of the *first* moment; higher moments follow from
            MOMENT_DECAY_RATIO and are therefore geometrically smaller.
        x0, sigma : float
            center and width of the Gaussian pulse. The defaults assume the
            domain x in [0, 1] used by the shipped configurations.

        Returns
        -------
        initial_values : np.ndarray
            conserved state vector of length `order + 2`.
        """
        if order < 0:
            raise ValueError(f"Moment order must be non-negative, got {order}.")

        initial_values = np.zeros(
            self.compute_number_of_variables(order),
            dtype = np.float64,
        )

        # A simple Gaussian pulse
        pulse = np.exp(-((position - x0) / sigma) ** 2)

        # Primitive variables
        h = h_ref + h_amp * pulse
        u_m = u_ref + u_amp * pulse

        # Conserved height and mean-momentum variables
        initial_values[0] = h
        initial_values[1] = h * u_m

        # Moment rows alpha_1 .. alpha_N, seeded on the geometric ray. The loop
        # replaces the old `if order > 0 / if order > 1` ladder and simply does
        # not execute at N = 0.
        for i in range(1, order + 1):
            alpha_i = alpha_amp * MOMENT_DECAY_RATIO ** (i - 1) * pulse
            initial_values[i + 1] = h * alpha_i

        return initial_values

    def _smooth_nested_profile_pulse_aggressive(
            self,
            order : int,
            position : float,
    ) -> np.ndarray:
        """
        Smooth custom benchmark for comparing recharge models across moment
        orders, in the strongly perturbed variant.

        Primitive design:
            h(x) = 1.0 + 0.05 * pulse
            u_m(x) = 1.0 + 0.15 * pulse
            alpha_i(x) = 0.10 * (-0.5)**(i-1) * pulse, for i = 1 .. N

        i.e. alpha_1 = 0.10, alpha_2 = -0.05, alpha_3 = 0.025, ... - the N <= 2
        values are unchanged from before the generic-N rewrite.

        Returned in conserved variables:
            [h, h*u_m, h*alpha_1, ..., h*alpha_N]
        """
        # Assumes the current config domain x in [0, 1]
        return self._nested_moment_pulse(
            order,
            position,
            h_ref = 1.0,
            h_amp = 0.05,
            u_ref = 1.0,
            u_amp = 0.15,
            alpha_amp = 0.10,
        )

    def _smooth_nested_profile_pulse_mild(
            self,
            order : int,
            position : float,
    ) -> np.ndarray:
        """
        Smooth custom benchmark for comparing recharge models across moment
        orders, in the weakly perturbed variant.

        Primitive design:
            h(x) = 1.0 + 0.03 * pulse
            u_m(x) = 0.8 + 0.08 * pulse
            alpha_i(x) = 0.04 * (-0.5)**(i-1) * pulse, for i = 1 .. N

        i.e. alpha_1 = 0.04, alpha_2 = -0.02, alpha_3 = 0.01, ... - the N <= 2
        values are unchanged from before the generic-N rewrite.

        Returned in conserved variables:
            [h, h*u_m, h*alpha_1, ..., h*alpha_N]
        """
        # Assumes the current config domain x in [0, 1]
        return self._nested_moment_pulse(
            order,
            position,
            h_ref = 1.0,
            h_amp = 0.03,
            u_ref = 0.8,
            u_amp = 0.08,
            alpha_amp = 0.04,
        )

    def _horton_moment_order_pulse(
            self,
            order : int,
            position : float,
    ) -> np.ndarray:
        """
        Smooth initial condition for comparing recharge models across moment
        orders under the same Horton rainfall-infiltration forcing.

        This initial condition is intended for the numerical test in which the
        recharge model is run at several moment orders N. The purpose of the
        test is to check whether the rainfall-infiltration source terms behave
        consistently across the moment hierarchy, while also allowing the
        higher-order models to evolve a non-trivial vertically resolved
        velocity profile.

        The construction is nested across moment order:
            N=0 : h(x) and u_m(x) are initialized.
            N=1 : the same h(x) and u_m(x) are used, and a non-zero first moment
            alpha_1(x) is added.
            N=2 : the same h(x), u_m(x), alpha_1(x) are used, and a non-zero
            second moment alpha_2(x) is added.
            N>2 : likewise, each further order adds one moment and changes
            nothing below it.

        Primitive variable design:
            h(x) = 1.0 + 0.05 * pulse(x)
            u_m(x) = 0.5 + 0.10 * pulse(x)
            alpha_i(x) = 0.10 * (-0.5)**(i-1) * pulse(x), for i = 1 .. N

        where pulse(x) is a smooth Gaussian perturbation centered at x = 0.5.
        The default parameter values assume the domain x in [0, 1].

        The former `N in {0, 1, 2}` restriction is gone: arbitrary N follows
        from the generic shifted-Legendre coefficient engine (RESTRUCTURE_PLAN.md
        Step 3), and the moments are seeded on a ray that stays outside the
        documented N=2 non-hyperbolic wedge - see MOMENT_DECAY_RATIO above.

        **A safe start is not a safe run, and this initial condition can only
        promise the first.** Measured on this case at N=4 (smoke_test_1 settings,
        200 cells, t_end = 1.0), comparing the two closures cell by cell:

            closure   non-hyperbolic cells   max|Im(lambda)|
            SWME      10697 / 124800         8.1e-2
            HSWME         0 / 124000         5.1e-25

        Both start clean - 0 of 200 cells at t = 0, which is the part the seeded
        ray controls - but under SWME the rainfall/infiltration forcing walks the
        state off that ray within the first stored step and never returns. That
        is the documented N >= 2 behavior of the closure (RESTRUCTURE_PLAN.md
        Section 6 measures ~21 % of state space at N=4), not something an initial
        condition can prevent. Both runs nonetheless stayed physical here
        (final h in [0.89, 1.04], all finite).

        So: `hyperbolic: true` (HSWME) is the right closure for N >= 2 runs of
        this case, as already used in `smoke_test_1.yaml` and `smoke_test_3.yaml`.
        """
        # Assumes the test uses x in [0, 1]
        return self._nested_moment_pulse(
            order,
            position,
            h_ref = 1.0,
            h_amp = 0.05,
            u_ref = 0.5,
            u_amp = 0.10,
            alpha_amp = 0.10,
        )
