from abc import ABC, abstractmethod
import numpy as np

from . import coefficients
from . import source_terms
from .topography import TopographySettings
from .wetdry import (
    DEFAULT_THRESHOLDS,
    WetDryThresholds,
    desingularized_primitives,
    desingularized_primitives_array,
)


def _compute_system_matrix_generic(
    order: int,
    values: np.ndarray,
    g: float = 1,
    hyperbolic: bool = False,
    thresholds: WetDryThresholds = DEFAULT_THRESHOLDS,
) -> np.ndarray:
    """Generic-N system matrix A(U). Backs `SWME1D.compute_system_matrix`,
    which validates its input and then delegates here.

    Replaced the former hardcoded `if order == 0/1/.../6:` blocks, which
    capped the model at N=6 and contained transcription errors at N=6.
    Behavior is pinned by tests/test_pde_regression.py against reference
    values captured from those blocks before they were deleted
    (RESTRUCTURE_PLAN.md Steps 2/3).

    Formulas (thesis Appendix C, general N):
        A[0,1] = 1
        A[1,0] = g*h - u_m^2 - sum_i alpha_i^2/(2i+1)
        A[1,1] = 2*u_m
        A[1,2:] = 2*alpha_i/(2i+1)
        A[2:,0] = -2*u_m*alpha - einsum('ijk,j,k->i', A_t, alpha, alpha)
        A[2:,1] = 2*alpha
        A[2:,2:] = u_m*I + einsum('ilk,k->il', 2*A_t + B_t, alpha)
    where A_t, B_t are the moment-moment blocks (indices 1..N) of the A_ijk,
    B_ijk tensors from coefficients.get_coefficients(N). The `hyperbolic` flag
    reproduces the existing HSWME regularization: alpha_2..alpha_N are zeroed
    (alpha_1 is kept) before assembling the matrix, matching the legacy
    per-order code's `if self.hyperbolic: alpha2 = 0; alpha3 = 0; ...`.
    """
    values = np.asarray(values, dtype=np.float64)
    n = order + 2
    Amat = np.zeros((n, n), dtype=np.float64)

    # Primitives come from the shared wet-dry rule, so the transport matrix,
    # the wave speeds and the source terms all agree about what a nearly-dry
    # cell means (RESTRUCTURE_PLAN.md Step 6). For h >= h_wet this is exactly
    # values[1]/h and values[2:]/h, i.e. unchanged from before wet-dry.
    h, um, alpha = desingularized_primitives(order, values, thresholds)

    if order == 0:
        Amat[0, 1] = 1.0
        Amat[1, 0] = g * h - um * um
        Amat[1, 1] = 2.0 * um
        return Amat

    # `alpha` is always a freshly allocated array, so the hyperbolic branch
    # can zero in place without a defensive copy.
    if hyperbolic and order > 1:
        alpha[1:] = 0.0  # zero alpha_2..alpha_N, keep alpha_1

    c = coefficients.get_coefficients(order)
    alpha_over_w = alpha * c.inv_two_i_plus_1

    # `A_m @ alpha @ alpha` and `transport_m @ alpha` are the einsum
    # contractions 'ijk,j,k->i' and 'ilk,k->il'; written as dots because they
    # are BLAS-backed and markedly cheaper than einsum at the small array
    # sizes and very high call frequency of this hot path.
    Amat[0, 1] = 1.0
    Amat[1, 0] = g * h - um * um - alpha @ alpha_over_w
    Amat[1, 1] = 2.0 * um
    Amat[1, 2:] = 2.0 * alpha_over_w
    Amat[2:, 0] = -2.0 * um * alpha - (c.A_m @ alpha) @ alpha
    Amat[2:, 1] = 2.0 * alpha
    moment_block = c.transport_m @ alpha
    # += um on the diagonal, without np.diag_indices' per-call arange
    moment_block.flat[:: order + 1] += um
    Amat[2:, 2:] = moment_block
    return Amat


def _compute_augmented_system_matrix_generic(
    order: int,
    augmented_values: np.ndarray,
    g: float = 1,
    hyperbolic: bool = False,
    thresholds: WetDryThresholds = DEFAULT_THRESHOLDS,
) -> np.ndarray:
    """Augmented-state system matrix A~(W) for W = (U, Z), generic in N.

    Bottom topography enters this solver as an extra, frozen path coordinate
    rather than as a cell-centered source term (RESTRUCTURE_PLAN.md Step 5).
    The scheme is already a Castro-Pares path-conservative fluctuation solver
    - `spatialDiscretization.PVM.compute_fluctuation` integrates A(psi(s))
    along the linear path between two cells - so the bed-slope non-conservative
    product g*h*dZ/dx is handled by the very same machinery as the moment
    transport, simply by widening the state:

        A~(W) = [ A(U)   g*h*e_momentum ]        (n+1) x (n+1),  n = order+2
                [   0           0       ]

    The last row is zero because Z does not evolve (dZ/dt = 0), and the last
    column is nonzero only in the momentum row: the momentum equation is

        d_t(h*u) + d_x(h*u^2 + g*h^2/2 + moment terms) + g*h*d_x(Z) = 0,

    so the coefficient multiplying d_x(Z) is +g*h. (Note the sign: an earlier
    revision of RESTRUCTURE_PLAN.md sketched this entry as -g*h, which is the
    sign the bed-slope term carries when written on the *right-hand* side as a
    source. Moved to the left-hand side, inside the transport matrix, it is
    +g*h. The C-property check below is what pins this down, and
    tests/test_topography.py asserts it directly.)

    Why this is well balanced. For a lake at rest - h + Z = H constant,
    u_m = alpha_i = 0 - the linear path between two neighbouring cells stays a
    lake at rest at every quadrature node, and

        A~(W(s)) . (W_R - W_L)
            row 0        : 1 * delta(h*u_m)                     = 0
            row momentum : g*h(s)*delta(h) + g*h(s)*delta(Z)
                         = g*h(s)*delta(h + Z)                  = 0
            rows moments : all coefficients vanish at rest       = 0

    identically, for every quadrature node and hence for the quadrature sum.
    The central part of the fluctuation is therefore exactly zero. Whether the
    *whole* fluctuation vanishes additionally depends on the numerical
    viscosity Q: it must annihilate the same jump. Q = |A~| (Roe) and the Osher
    variant do, since the jump lies in ker(A~); Q = c*I + ... (LF, PRICE) does
    not, because of the constant term. See the `well_balanced` flag on the
    schemes in `spatialDiscretization.py`, which `ClassicalSimulation1D` warns
    about.

    Parameters
    ----------
    order : int
        order of the moment model
    augmented_values : np.ndarray
        1D augmented state [h, h*u_m, h*a_1, ..., h*a_N, Z], length order+3
    g : float
        gravitational constant
    hyperbolic : bool
        HSWME regularization flag, forwarded to the base matrix
    thresholds : WetDryThresholds
        wet-dry thresholds, forwarded to the base matrix

    Returns
    -------
    A_tilde : np.ndarray
        shape (order+3, order+3)
    """
    augmented_values = np.asarray(augmented_values, dtype=np.float64)
    n = order + 2
    A_tilde = np.zeros((n + 1, n + 1), dtype=np.float64)
    A_tilde[:n, :n] = _compute_system_matrix_generic(
        order,
        augmented_values[:n],
        g=g,
        hyperbolic=hyperbolic,
        thresholds=thresholds,
    )
    A_tilde[1, n] = g * augmented_values[0]
    return A_tilde


#TODO: implement MomentModel as a subclass of PDE and include the possibility of simulating PDEs that are not moment models (and don't have an order)
class PDE(ABC):
    """
    This interface represents a partial differential equation.

    ...

    Attributes
    ----------
    initial_condition : str
        initial condition for the partial differential equation

    
    Abstract methods
    -------
    def init(self):
        initializes the pde object
    def compute_system_matrix(self,order,values):
        computes the system matrix of the partial differential equation evaluated in the given values, for the given order.
    def compute_source_term(self,order,values,delta_t):
        computes the source term of the partial differential equation evaluated in the given values, for the given order.
    def get_initial_values(self,order,initial_condition,position):
        calculates the initial values for one specific physical position
    def compute_number_of_variables(self,order):
        computes the number of state variables in the PDE given the order of the moment model
    def compute_max_wavespeed(self,order,values):
        compute the maximum wavespeed in the system, used to calculate a timestep that satisfies CFL condition
    def convert_to_primitive(self,order,data_matrix_convective):
        converts the computed values to the values of the primitive variables
    """

    @abstractmethod
    def __init__(self):
        """
        Implemented in the child classes
        """

        pass

    @abstractmethod
    def compute_system_matrix(self,
                              order: int,
                              values: np.ndarray) -> np.ndarray:
        """
        Computes the system matrix of the PDE model.

        Parameters
        ----------
        order : int
            order of the moment model PDE (TODO: create MomentModel as a subclass of PDE)
        values : numpy 1D array
            values of the variables

        
        Returns
        -------
        A: np.ndarray
            system matrix

        """

        pass
    
    @abstractmethod
    def compute_source_term(self,
                            order: int,
                            values: np.ndarray,
                            delta_t: float) -> np.ndarray:
        """
        Computes the source term with a given order of the PDE evaluated in the given values.

        Parameters
        ----------
        order : int
            order of the moment model PDE (TODO: create MomentModel as a subclass of PDE)
        values : numpy 1D array
            values of the variables
        delta_t : float
            current time step. Needed by implementations whose source term is
            time-step dependent: the linear/implicit branch returns an already
            solved (I - delta_t*S)^-1 matrix, and the recharge extension passes
            delta_t to its infiltration closures (available-water limiting).

        Returns
        -------
        S: numpy 1D array or numpy 2D array
            source term vector if the moment model has a non-linear source term,
            source term matrix if the moment model has a linear source term

        Notes
        -----
        Returning a *matrix* is only valid when the implementation is paired
        with `timeIntegration.Implicit`, which multiplies it onto the state;
        `SWME1D` guards this with a constructor-time check on `linear_source`.
        """

        pass

    @abstractmethod
    def get_initial_values(self,
                           order: int,
                           initial_condition: str,
                           position) -> np.ndarray:

        """
        calculates the initial values for one specific physical position

        Parameters
        ----------
        order : int
            order of the moment model PDE (TODO: create MomentModel as a subclass of PDE)
        initial condition : str
            name of the initial condition
        position : float (if 1D) or numpy 1D array of floats (2D)
            the physical position in which the initial values are computed
        
        
        Returns
        -------
        initial_values: numpy 1D array (if 1D) or numpy 2D array (if 2D)
            initial values for the given initial condition evaluated in the phyiscal position

        """

        pass

    @abstractmethod
    def compute_number_of_variables(self,
                           order: int) -> int:

        """
        given the order of the moment model expansion, compute the number of state variables in the PDE

        Parameters
        ----------
        order : int
            order of the moment model PDE (TODO: create MomentModel as a subclass of PDE)
        
        
        Returns
        -------
        number_of_variables: int
            number of state variables in the PDE

        """

        pass

    @abstractmethod
    def compute_max_wavespeed(self,
                           order: int,
                           values: np.ndarray) -> float:
        """
        Computes the maximum wavespeed magnitude in the system.
        It effectively computes (or approximates) the maximum eigenvalue (in absolute value) of the system matrix.
        This is done by providing an analytical function instead of solving an expensive eigenvalue problem.

        Parameters
        ----------
        order : int
            order of the moment model PDE (TODO: create MomentModel as a subclass of PDE)
        values : numpy 1D array
            values of the variables

        
        Returns
        -------
        max_wavespeed: float
            Maximum wave speed (in absolute value) appearing in the PDE system

        """

        pass

    @abstractmethod
    def convert_to_primitive(self,
                           order: int,
                           data_matrix_convective: np.ndarray) -> np.ndarray:
        """
        Converts the computed variables to primitive variables

        Parameters
        ----------
        order : int
            order of the moment model PDE
        data_matrix_convective : numpy array
            array containing all the computed values 
        
        Returns
        -------
        data_matrix_primitive : numpy array
            array containing all the values of the primitive variables

        """

class SWME1D(PDE):

    """
    This class represents the one-dimensional Shallow Water Moment Equations (SWME1D).

    ...

    Attributes
    ----------
    initial_condition : str
        initial condition for the SWME1D
    viscosity : float
        value for the dynamic viscosity
    slip_length : float
        value for the slip length
    hyperbolic : boolean
        whether the model is hyperbolic, true (HSWME) or false (SWME)
    linear_source : boolean
        true if the source term is represented as a constant matrix multiplied by the state vector,
        false if the source term is represented in vector form

    
    Implemented methods from interface PDE
    ---------------------------------
    def __init__(self, initial_condition, viscosity, slip_length, hyperbolic, linear_source, topography, wet_dry):
        initializes the SWME1D object
    def compute_system_matrix(self,order,values):
        computes the system matrix of the partial differential equation evaluated in the given values, for the given order.
    def compute_source_term(self,order,values,delta_t):
        computes the source term of the partial differential equation evaluated in the given values, for the given order.
    def get_initial_values(self,order,initial_condition,position):
        calculates the initial values for one specific physical position
    def compute_number_of_variables(self,order):
        computes the number of state variables in the PDE given the order of the moment model
    def compute_max_wavespeed(self,order,values):
        compute the maximum wavespeed in the system, used to calculate a timestep that satisfies CFL condition
    def convert_to_primitive(self,order,data_matrix_convective):
        converts the computed values to the values of the primitive variables


    Instance methods
    ----------------
    def compute_vertical_velocity_profile(self,values):
        reconstruct the vertical velocity profiles from the moment values
    def _compute_source_matrix_inverse(self,order,values,delta_t,g = 1):
        computes a matrix for efficient implicit numerical solution
    """

    def __init__(self,
                initial_condition: str,
                viscosity: float,
                slip_length: float,
                hyperbolic: bool,
                linear_source: bool,
                topography: TopographySettings | None = None,
                wet_dry: WetDryThresholds | None = None):
        """
        Constructs all the necessary attributes for the SWME1D object.

        Parameters
        ----------
        initial_condition : str
            initial condition of the PDE
        viscosity : float
            dynamic viscosity value
        slip_length : float
            slip length value
        hyperbolic : boolean
            true if hyperbolic, false if not hyperbolic
        linear_source : boolean
            true if the source term is represented as a constant matrix multiplied by the state vector,
            false if the source term is represented in vector form
        topography : TopographySettings, optional
            bed-elevation settings, needed only by the topography-aware
            initial conditions ('lakeAtRest', 'perturbedLakeAtRest'), which
            cannot be expressed as a function of position alone. Defaults to a
            flat bed at zero, i.e. exactly the pre-topography behavior. The
            bed's effect on the *dynamics* does not come through here - it
            comes through `compute_augmented_system_matrix` and the sampled
            elevation stored on the mesh.
        wet_dry : WetDryThresholds, optional
            thresholds governing how nearly-dry cells are handled - see
            `swme.wetdry`. Defaults to h_dry = 1e-4, h_wet = 1e-3, sized for
            the thesis' h ~ O(1) test cases; scale them to the problem. Runs
            that stay above h_wet everywhere are unaffected by this entirely.
        """
        self.initial_condition = initial_condition
        self.viscosity = viscosity
        self.slip_length = slip_length
        self.hyperbolic = hyperbolic
        self.linear_source = linear_source
        self.exact_source_computation = False
        self.topography = topography if topography is not None else TopographySettings()
        self.wet_dry = wet_dry if wet_dry is not None else DEFAULT_THRESHOLDS

    def compute_system_matrix(self,
                              order: int,
                              values: np.ndarray,
                              g = 1) -> np.ndarray:
        """
        Computes the system matrix A(U) of the SWME/HSWME model, generic in
        the moment order N.

        Implemented via the shifted-Legendre projection tensors from
        `swme.coefficients`; see `_compute_system_matrix_generic` (module
        level, above) for the formulas. This replaces the former hardcoded
        `if order == 0/1/.../6:` blocks, which capped the model at N=6 and
        contained transcription errors in the N=6 block (see
        RESTRUCTURE_PLAN.md Step 2/3 and tests/test_pde_regression.py).

        Parameters
        ----------
        order : int
            order of the moment model
        values : np.ndarray
            1D conserved state vector [h, h*u_m, h*a_1, ..., h*a_N]
        g : float
            gravitational constant (1 by default, since the simulations are
            based on dimensionless equations)

        Returns
        -------
        A : np.ndarray
            system matrix, shape (order+2, order+2)
        """
        values = np.asarray(values, dtype = np.float64)

        if values.ndim != 1:
            raise ValueError(
                f"Expected 1D state vector, got shape {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite state encountered in compute_system_matrix: {values}"
            )
        # h == 0 is a legitimate state since wet-dry landed (Step 6): a dry
        # cell is dry, not an error. Only a *negative* height is impossible -
        # the positivity-preserving timestep limiter in
        # ClassicalSimulation1D.run_simulation exists to guarantee it cannot
        # happen, so this stays as the assertion that it worked.
        if values[0] < 0.0:
            raise ValueError(
                f"Negative height h={values[0]} in compute_system_matrix, "
                f"values={values}"
            )

        return _compute_system_matrix_generic(
            order,
            values,
            g = g,
            hyperbolic = self.hyperbolic,
            thresholds = self.wet_dry,
        )

    def compute_augmented_system_matrix(self,
                                        order: int,
                                        augmented_values: np.ndarray,
                                        g = 1) -> np.ndarray:
        """
        Computes the augmented system matrix A~(W), W = (U, Z), that couples
        bottom topography into the path-conservative scheme.

        See `_compute_augmented_system_matrix_generic` (module level, above)
        for the formula, the sign of the bed-slope entry, and the argument for
        why this recovers the C-property. Used by
        `ClassicalSimulation1D.run_simulation` in place of
        `compute_system_matrix` whenever the mesh carries a non-flat bed.

        Parameters
        ----------
        order : int
            order of the moment model
        augmented_values : np.ndarray
            1D augmented state [h, h*u_m, h*a_1, ..., h*a_N, Z], length
            order+3
        g : float
            gravitational constant (1 by default; the equations are
            dimensionless)

        Returns
        -------
        A_tilde : np.ndarray
            augmented system matrix, shape (order+3, order+3)
        """
        augmented_values = np.asarray(augmented_values, dtype = np.float64)

        expected = order + 3
        if augmented_values.ndim != 1 or augmented_values.shape[0] != expected:
            raise ValueError(
                f"Expected a 1D augmented state of length {expected} "
                f"([h, h*u_m, h*a_1..a_{order}, Z]), got shape "
                f"{augmented_values.shape}."
            )
        if not np.isfinite(augmented_values).all():
            raise ValueError(
                "Non-finite state encountered in "
                f"compute_augmented_system_matrix: {augmented_values}"
            )
        # See compute_system_matrix: dry (h == 0) is allowed, negative is not.
        if augmented_values[0] < 0.0:
            raise ValueError(
                f"Negative height h={augmented_values[0]} in "
                f"compute_augmented_system_matrix, values={augmented_values}"
            )

        return _compute_augmented_system_matrix_generic(
            order,
            augmented_values,
            g = g,
            hyperbolic = self.hyperbolic,
            thresholds = self.wet_dry,
        )

    def compute_source_term(self,
                            order: int,
                            values: np.ndarray,
                            delta_t: float,
                            g = 1) -> np.ndarray:
        """
        Computes the source term of the SWME/HSWME model, generic in the
        moment order N.

        For the base model the source is the Navier-slip bed friction alone,
        i.e. thesis eq. (3.38) with R = I = f_R = f_I = 0, so this returns
        -P_slip(U). Rainfall/infiltration mass exchange and mixing friction
        are additions made by the recharge extension and live in
        `recharge.source_terms`, not here.

        Parameters
        ----------
        order : int
            order of the moment model
        values : np.ndarray
            1D conserved state vector [h, h*u_m, h*a_1, ..., h*a_N]
        delta_t : float
            current time step (used only by the linear/implicit branch)
        g : float
            gravitational constant (unused here; kept for interface symmetry)

        Returns
        -------
        S : np.ndarray
            source term vector, or - when `self.linear_source` is set - the
            already-solved implicit-Euler matrix (I - delta_t*S(h))^-1, which
            `timeIntegration.Implicit` multiplies onto the state (see
            `_compute_source_matrix_inverse`).
        """
        if self.linear_source:
            return self._compute_source_matrix_inverse(order, values, delta_t)

        return -source_terms.compute_navier_slip_friction(
            order,
            values,
            self.viscosity,
            self.slip_length,
        )

    def _compute_source_matrix_inverse(self,
                                      order: int,
                                      values: np.ndarray,
                                      delta_t,
                                      g = 1) -> np.ndarray:
        """
        The friction step in the SWME can be written as w'(t) = S(h0).w(t).
        Implicit Euler is then w^{n+1} = (I - delta_t*S(h0))^{-1}.w^n, and
        this function returns that matrix.

        Generic in N: S(h0) is assembled from the shifted-Legendre projection
        tensors by `swme.source_terms.compute_friction_operator_matrix`, and
        (I - delta_t*S) is inverted numerically at runtime. This replaces
        ~1700 lines of per-order closed forms that had been precomputed in
        Wolfram Mathematica and capped the model at N=6 (RESTRUCTURE_PLAN.md
        Step 3). The cost is one small (order+2)x(order+2) inverse per cell
        per step, negligible next to the 5-point Gauss quadrature already
        performed per interface.

        Parameters
        ----------
        order : integer
            the order of the system
        values : np.ndarray
            array containing the current values in the grid cell; only the
            water height values[0] is read, because the friction operator
            depends on the state solely through h
        delta_t : float
            current time step
        g = 1 : float
            gravitational constant (unused; kept for interface symmetry)

        Returns
        -------
        S_inv : np.ndarray
            (I - delta_t * S(h0))^{-1}, shape (order+2, order+2)
        """
        n = self.compute_number_of_variables(order)
        S = source_terms.compute_friction_operator_matrix(
            order,
            values[0],
            self.viscosity,
            self.slip_length,
        )
        return np.linalg.inv(np.eye(n) - delta_t * S)

    def get_initial_values(self,
                           order: int,
                           initial_condition: str,
                           position: float) -> np.ndarray:
        initial_values = np.zeros(self.compute_number_of_variables(order))
        if initial_condition == 'constantHeight_noVelocity':
            initial_values[0] = 1
            initial_values[1] = 0
            if order > 0:
                initial_values[2] = 0 
            if order > 1:
                initial_values[3] = 0 
            if order > 2:
                initial_values[4] = 0 
            if order > 3:
                initial_values[5] = 0 
            if order > 4:
                initial_values[6] = 0 
            if order > 5:
                initial_values[7] = 0
        elif initial_condition == 'constantHeight_constantVelocity':
            initial_values[0] = 1
            initial_values[1] = 1*initial_values[0]
            if order > 0:
                initial_values[2] = 0 
            if order > 1:
                initial_values[3] = 0 
            if order > 2:
                initial_values[4] = 0 
            if order > 3:
                initial_values[5] = 0 
            if order > 4:
                initial_values[6] = 0 
            if order > 5:
                initial_values[7] = 0
        elif initial_condition == 'damBreak_noVelocity':
            x0 = 0
            if position < x0:
                initial_values[0] = 2
                initial_values[1] = 0*initial_values[0]
                if order > 0:
                    initial_values[2] = 0 
                if order > 1:
                    initial_values[3] = 0 
                if order > 2:
                    initial_values[4] = 0 
                if order > 3:
                    initial_values[5] = 0 
                if order > 4:
                    initial_values[6] = 0
                if order > 5:
                    initial_values[7] = 0 
            else:
                initial_values[0] = 1
                initial_values[1] = 0*initial_values[0]
                if order > 0:
                    initial_values[2] = 0 
                if order > 1:
                    initial_values[3] = 0 
                if order > 2:
                    initial_values[4] = 0 
                if order > 3:
                    initial_values[5] = 0 
                if order > 4:
                    initial_values[6] = 0 
                if order > 5:
                    initial_values[7] = 0
        elif initial_condition == 'damBreak_constantVelocity':
            x0 = 0
            if position < x0:
                initial_values[0] = 3
                initial_values[1] = 0.25*initial_values[0]
                if order > 0:
                    initial_values[2] = 0 
                if order > 1:
                    initial_values[3] = 0 
                if order > 2:
                    initial_values[4] = 0 
                if order > 3:
                    initial_values[5] = 0 
                if order > 4:
                    initial_values[6] = 0
                if order > 5:
                    initial_values[7] = 0 
            else:
                initial_values[0] = 1
                initial_values[1] = 0.25*initial_values[0]
                if order > 0:
                    initial_values[2] = 0 
                if order > 1:
                    initial_values[3] = 0 
                if order > 2:
                    initial_values[4] = 0 
                if order > 3:
                    initial_values[5] = 0 
                if order > 4:
                    initial_values[6] = 0 
                if order > 5:
                    initial_values[7] = 0
        elif initial_condition == 'linearHeight_noVelocity':
            initial_values[0] = 1 + 0.1*position
            initial_values[1] = 0*initial_values[0]
            if order > 0:
                initial_values[2] = 0 
            if order > 1:
                initial_values[3] = 0 
            if order > 2:
                initial_values[4] = 0 
            if order > 3:
                initial_values[5] = 0 
            if order > 4:
                initial_values[6] = 0 
            if order > 5:
                initial_values[7] = 0
            if order > 6:
                initial_values[8] = 0
        elif initial_condition == 'smooth_wave':
            initial_values[0] = 1 + 0.5*np.exp(-5*position**2)
            initial_values[1] = 1.0*initial_values[0]
            if order > 0:
                initial_values[2] = 1.0 
            if order > 1:
                initial_values[3] = 1.0 
            if order > 2:
                initial_values[4] = 1.0 
            if order > 3:
                initial_values[5] = 1.0 
            if order > 4:
                initial_values[6] = 1.0 
            if order > 5:
                initial_values[7] = 1.0  
        elif initial_condition == 'smooth_constantVelocity':
            initial_values[0] = 1 + 0.5*np.exp(-15*position**2)
            initial_values[1] = 0.2*initial_values[0]
            if order > 0:
                initial_values[2] = 0
            if order > 1:
                initial_values[3] = 0 
            if order > 2:
                initial_values[4] = 0 
            if order > 3:
                initial_values[5] = 0 
            if order > 4:
                initial_values[6] = 0 
            if order > 5:
                initial_values[7] = 0  
        elif initial_condition == 'symmetric_damBreak':
            x0 = -2
            x1 = 2
            if x0 < position < x1:
                initial_values[0] = 2
                initial_values[1] = 0.*initial_values[0]
                if order > 0:
                    initial_values[2] = 0 
                if order > 1:
                    initial_values[3] = 0 
                if order > 2:
                    initial_values[4] = 0 
                if order > 3:
                    initial_values[5] = 0 
                if order > 4:
                    initial_values[6] = 0
                if order > 5:
                    initial_values[7] = 0 
            else:
                initial_values[0] = 1
                initial_values[1] = 0.*initial_values[0]
                if order > 0:
                    initial_values[2] = 0 
                if order > 1:
                    initial_values[3] = 0 
                if order > 2:
                    initial_values[4] = 0 
                if order > 3:
                    initial_values[5] = 0 
                if order > 4:
                    initial_values[6] = 0 
                if order > 5:
                    initial_values[7] = 0
        elif initial_condition == 'smooth_plus_damBreak':
            x0 = -7
            x1 = 7
            if  position < x0:
                initial_values[0] = 4
                initial_values[1] = 0.05*initial_values[0]
                if order > 0:
                    initial_values[2] = -0.01*initial_values[0] 
                if order > 1:
                    initial_values[3] = 0 
                if order > 2:
                    initial_values[4] = 0*initial_values[0] 
                if order > 3:
                    initial_values[5] = 0 
                if order > 4:
                    initial_values[6] = 0
                if order > 5:
                    initial_values[7] = 0 
            else:
                initial_values[0] = 3 + np.exp(-1.5*(position-x1)**2)
                initial_values[1] = 0.05*initial_values[0]
                if order > 0:
                    initial_values[2] = -0.01*initial_values[0] 
                if order > 1:
                    initial_values[3] = 0 
                if order > 2:
                    initial_values[4] = 0*initial_values[0] 
                if order > 3:
                    initial_values[5] = 0 
                if order > 4:
                    initial_values[6] = 0 
                if order > 5:
                    initial_values[7] = 0 
        elif initial_condition == 'linearDamBreak_noVelocity':
            x0 = -4
            x1 = 4
            if x0 < position < x1:
                initial_values[0] = 2 + (position+4)/8.0
                initial_values[1] = 0*initial_values[0]
                if order > 0:
                    initial_values[2] = 0 
                if order > 1:
                    initial_values[3] = 0 
                if order > 2:
                    initial_values[4] = 0 
                if order > 3:
                    initial_values[5] = 0 
                if order > 4:
                    initial_values[6] = 0
                if order > 5:
                    initial_values[7] = 0 
            else:
                initial_values[0] = 2
                initial_values[1] = 0*initial_values[0]
                if order > 0:
                    initial_values[2] = 0 
                if order > 1:
                    initial_values[3] = 0 
                if order > 2:
                    initial_values[4] = 0 
                if order > 3:
                    initial_values[5] = 0 
                if order > 4:
                    initial_values[6] = 0 
                if order > 5:
                    initial_values[7] = 0 
        elif initial_condition == 'colliding_damBreak':
            x0 = -0.5
            x1 = 0.5
            if position < x0 or position > x1:
                initial_values[0] = 3
                initial_values[1] = 0.5*initial_values[0]
                if order > 0:
                    initial_values[2] = 0 
                if order > 1:
                    initial_values[3] = 0 
                if order > 2:
                    initial_values[4] = 0 
                if order > 3:
                    initial_values[5] = 0 
                if order > 4:
                    initial_values[6] = 0
                if order > 5:
                    initial_values[7] = 0 
            else:
                initial_values[0] = 1
                initial_values[1] = 0.5*initial_values[0]
                if order > 0:
                    initial_values[2] = 0 
                if order > 1:
                    initial_values[3] = 0 
                if order > 2:
                    initial_values[4] = 0 
                if order > 3:
                    initial_values[5] = 0 
                if order > 4:
                    initial_values[6] = 0 
                if order > 5:
                    initial_values[7] = 0
        elif initial_condition == 'damBreak_dryBed':
            # The canonical wet-dry benchmark (RESTRUCTURE_PLAN.md Step 6):
            # a column of water at rest released onto a completely dry bed.
            # The dry side is *exactly* zero, not a thin film - the point is
            # that the scheme handles a genuine vacuum front, and a small
            # positive floor would quietly turn this into an easier problem.
            initial_values[0] = 1.0 if position < 0.0 else 0.0
        elif initial_condition == 'damBreak_dryBed_partial':
            # Partial dam break: a shallow but nonzero downstream layer, so
            # the front is a wetting front rather than a vacuum front. Both
            # are worth testing; this one has an exact Stoker solution.
            initial_values[0] = 1.0 if position < 0.0 else 0.1
        elif initial_condition in ('lakeAtRest', 'perturbedLakeAtRest'):
            # Topography-aware initial conditions (RESTRUCTURE_PLAN.md Step 5).
            # Unlike every other case above, these are not a function of
            # position alone - they need the bed profile, which arrives via
            # `self.topography` (see TopographySettings).
            #
            # 'lakeAtRest' is the C-property benchmark: a flat free surface
            # h + Z = H over an arbitrary bed, at rest. A well-balanced scheme
            # must hold it exactly, forever.
            # 'perturbedLakeAtRest' adds a Gaussian free-surface bump on top,
            # the standard follow-up test: the perturbation must propagate
            # without the bed generating spurious waves of the same order.
            settings = self.topography
            bed = settings.elevation_at(position)
            height = settings.reference_water_level - bed

            if initial_condition == 'perturbedLakeAtRest' and settings.perturbation_amplitude != 0.0:
                xi = (position - settings.perturbation_center) / settings.perturbation_width
                height += settings.perturbation_amplitude*np.exp(-xi*xi)

            if not np.isfinite(height) or height <= 0.0:
                raise ValueError(
                    f"Initial condition '{initial_condition}' produced a "
                    f"non-positive water height h={height} at x={position} "
                    f"(bed Z={bed}, reference level "
                    f"{settings.reference_water_level}). Raise the reference "
                    "level above the highest point of the bed - dry cells are "
                    "not supported yet (RESTRUCTURE_PLAN.md Step 6)."
                )

            # At rest: zero momentum and zero moments, at every order.
            initial_values[0] = height
        elif initial_condition == 'smooth_wave_smallHeightGradient':
            initial_values[0] = 1 + 0.5*np.exp(-3*position**2)
            initial_values[1] = 1.0*initial_values[0]
            if order > 0:
                initial_values[2] = 1.0 
            if order > 1:
                initial_values[3] = 1.0 
            if order > 2:
                initial_values[4] = 1.0 
            if order > 3:
                initial_values[5] = 1.0 
            if order > 4:
                initial_values[6] = 1.0 
            if order > 5:
                initial_values[7] = 1.0  
        return initial_values
    
    def compute_number_of_variables(self,
                                    order: int) -> int:
        number_of_variables = order + 2
        return int(number_of_variables)

    def compute_max_wavespeed(self,
                           order: int,
                           values: np.ndarray,
                           g=1) -> float:

        values = np.asarray(values, dtype = np.float64)

        if np.all(values[:,0] >= self.wet_dry.h_wet):
            # Fully-wet fast path, kept as the literal pre-wet-dry expression.
            # Not merely an optimization: `(a*a)/(h*h)` and `(a/h)**2` agree
            # mathematically but not to the last bit, and the wave speed sets
            # delta_t, so any difference here would perturb every subsequent
            # step. Keeping this branch verbatim is what lets runs that never
            # approach drying reproduce earlier results exactly.
            wave_speed_sqrt = values[:,0]*int(g)
            for i in range(order):
                wave_speed_sqrt += np.divide(values[:,i+2]*values[:,i+2],values[:,0]*values[:,0])
            max_wave_speed_plus = np.max(np.abs(np.divide(values[:,1],values[:,0])+np.sqrt(wave_speed_sqrt)))
            max_wave_speed_min = np.max(np.abs(np.divide(values[:,1],values[:,0])-np.sqrt(wave_speed_sqrt)))
            return max(max_wave_speed_plus,max_wave_speed_min)

        # Something on the grid is at or below h_wet: go through the shared
        # wet-dry rule, so a dry cell contributes a wave speed of 0 rather
        # than inf/NaN and cannot collapse the timestep.
        h, um, alpha = desingularized_primitives_array(order, values, self.wet_dry)

        wave_speed_sqrt = h*int(g)
        for i in range(order):
            wave_speed_sqrt = wave_speed_sqrt + alpha[:,i]*alpha[:,i]
        wave_speed_sqrt = np.maximum(wave_speed_sqrt, 0.0)

        max_wave_speed_plus = np.max(np.abs(um+np.sqrt(wave_speed_sqrt)))
        max_wave_speed_min = np.max(np.abs(um-np.sqrt(wave_speed_sqrt)))
        max_wavespeed = max(max_wave_speed_plus,max_wave_speed_min)

        # An entirely dry grid has no waves at all, so the CFL condition
        # genuinely imposes no constraint - but the caller divides by this to
        # get delta_t, so it must be finite and positive. Fall back to the
        # wave speed of the shallowest depth still considered wet.
        #
        # Be aware of what this means: on a completely dry domain the timestep
        # is no longer controlled by anything physical, and any source term
        # (rainfall, most obviously) is then integrated over a step chosen by
        # this fallback rather than by accuracy. Start such a case from a thin
        # film, or set t_end/resolution so the fallback step is small enough
        # for the source. CFL-based stepping cannot solve this on its own.
        if not np.isfinite(max_wavespeed) or max_wavespeed <= 0.0:
            return float(np.sqrt(self.wet_dry.h_wet * int(g)))
        return max_wavespeed

    def compute_vertical_velocity_profile(self,
                                          order: int,
                                          values: np.ndarray,
                                          z_points: np.ndarray) -> np.ndarray:
        """
        Reconstructs the vertical velocity profile from the moment values and
        evaluates it pointwise, for arbitrary moment order N:

            u(z) = u_m + sum_{i=1..N} a_i * phi_i(z)

        Generic in N via `swme.coefficients.eval_phi`, replacing a hardcoded
        per-order polynomial list that silently truncated above N=6
        (RESTRUCTURE_PLAN.md Step 4.5).

        Parameters
        ----------
        order: integer
            order of the model
        values: np.ndarray (2D)
            *Post-processed* (primitive) array, one row per mesh cell, with the
            position column prepended: [x, h, u_m, a_1, ..., a_N]. This is what
            `Simulation._post_processing` produces and what
            `plotting.SWME1DPlotClassical.plot` passes in - note the offset,
            u_m is column 2, not column 1.
        z_points:
            the locations in vertical direction in which the velocity is computed

        Returns
        -------
        velocity_profile: numpy 2D array
            lateral velocity evaluated in each point in z_points, per cell
        """
        values = np.asarray(values, dtype = np.float64)
        z_points = np.asarray(z_points, dtype = np.float64)

        expected_columns = order + 3   # x, h, u_m, a_1..a_N
        if values.ndim != 2 or values.shape[1] < expected_columns:
            raise ValueError(
                f"compute_vertical_velocity_profile expects a post-processed "
                f"array with at least {expected_columns} columns "
                f"([x, h, u_m, a_1..a_{order}]), got shape {values.shape}."
            )

        # phi has shape (order+1, len(z_points)); the moment coefficients
        # (u_m first, then a_1..a_N) live in columns 2 .. order+2.
        phi = coefficients.eval_phi(order, z_points)
        return values[:, 2:expected_columns] @ phi

    def convert_to_primitive(self,
                           order: int,
                           data_matrix_convective: np.ndarray) -> np.ndarray:
        
        data_matrix_convective = np.asarray(data_matrix_convective,
                                            dtype = np.float64)
        data_matrix_primitive = np.full(np.shape(data_matrix_convective),
                                        np.nan, dtype = np.float64)
        
        h = data_matrix_convective[:, 0]
        valid = np.isfinite(h) & (h > 0.0)

        # Always copy height
        data_matrix_primitive[:, 0] = h

        # Only divide where height is valid
        data_matrix_primitive[valid, 1] = (
            data_matrix_convective[valid, 1] / h[valid]
        )

        for j in range(order):
            data_matrix_primitive[valid, j + 2] = (
                data_matrix_convective[valid, j + 2] / h[valid]
            )

        # A dry cell (h == 0) is a legitimate state since Step 6, not missing
        # data, so report its primitives as zero rather than NaN - otherwise
        # every downstream mean/min/max in the CSV summaries would be poisoned
        # by the dry part of the domain. NaN is still reserved for genuinely
        # invalid states (non-finite or negative h), which should not occur.
        dry = np.isfinite(h) & (h >= 0.0) & ~valid
        data_matrix_primitive[dry, 1:] = 0.0

        return data_matrix_primitive
    
