from abc import ABC, abstractmethod
import numpy as np

from . import coefficients
from . import source_terms


def _compute_system_matrix_generic(
    order: int,
    values: np.ndarray,
    g: float = 1,
    hyperbolic: bool = False,
    eps_div: float = 1e-12,
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

    h = values[0]
    h_reg = h if h > eps_div else eps_div
    um = values[1] / h_reg

    if order == 0:
        Amat[0, 1] = 1.0
        Amat[1, 0] = g * h - um * um
        Amat[1, 1] = 2.0 * um
        return Amat

    # values[2:] / h_reg already allocates a fresh float64 array, so the
    # hyperbolic branch can zero in place without a defensive copy.
    alpha = values[2:] / h_reg
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
    def init(self, initial_condition,viscosity,slip_length,hyperbolic,linear_source,exact_source_computation):
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
                linear_source: bool):
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
        """
        self.initial_condition = initial_condition
        self.viscosity = viscosity
        self.slip_length = slip_length
        self.hyperbolic = hyperbolic
        self.linear_source = linear_source
        self.exact_source_computation = False

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
        if values[0] <= 0.0:
            raise ValueError(
                f"Non-positive height h={values[0]} in compute_system_matrix, "
                f"values={values}"
            )

        return _compute_system_matrix_generic(
            order,
            values,
            g = g,
            hyperbolic = self.hyperbolic,
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

        wave_speed_sqrt = values[:,0]*int(g)
        for i in range(order):
            wave_speed_sqrt += np.divide(values[:,i+2]*values[:,i+2],values[:,0]*values[:,0])
        max_wave_speed_plus = np.max(np.abs(np.divide(values[:,1],values[:,0])+np.sqrt(wave_speed_sqrt)))
        max_wave_speed_min = np.max(np.abs(np.divide(values[:,1],values[:,0])-np.sqrt(wave_speed_sqrt)))
        max_wavespeed = max(max_wave_speed_plus,max_wave_speed_min)

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

        return data_matrix_primitive 
    
