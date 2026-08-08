from abc import ABC, abstractmethod
import numpy as np
import scipy.optimize as spopt
from collections.abc import Callable

class TimeIntegration(ABC):

    """
    This abstract class represents a time integrator.

    ...

    Attributes
    ----------
    None

    
    Abstract methods
    -------
    def integrate(self):
        integrates the equation in time

    Instance methods
    ----------------
    def __init__(self):
        initializes the time integrator

    """

    def __init__(self):
        """
        Initializes the time integrator 

        """        
        pass

    @abstractmethod
    def integrate(self):

        """
        Documented in the child classes

        """
        pass

class Implicit(TimeIntegration):

    """
    This abstract class represents an implicit time integrator.

    ...

    Attributes
    ----------
    linear : boolean
        True if the source term can be written in linear form, false if the source term can not be written in linear form

    Implemented methods from abstract parent class TimeIntegration
    -------------
    def integrate(self,initial_value,rhs_f,delta_t):
        integrates the equation dw/dt = rhs_f(w) with a time step of delta_t

            
    Abstract methods    
    -----------------
    def _compute_residual(self,initial_value,rhs_f,delta_t):
        construct the residual function residual(x), the roots of which will be computed numerically

    Instance methods
    ----------------
    def __init__(self,linear):
        initializes the implicit time integrator

    """

    def __init__(self,
                 linear: bool):
        self.linear = linear

    def integrate(self,
                  initial_value: np.ndarray,
                  rhs_f: Callable[...,np.ndarray],
                  delta_t: float) -> np.ndarray:
        
        """
        integrates the equation dw/dt = rhs_f(w) starting from initial_value with a time step of delta_t with an implicit time integration method

        Parameters
        ----------
        initial_value: np.ndarray
            initial value w0
        rhs_f: function
            the right-hand side function describing the time evolution of w
        delta_t: float
            time step size
        
        Returns
        -------
        end_values: np.ndarray
            final values 

        """

        if self.linear:
            end_values = rhs_f(initial_value,delta_t)@initial_value
        else: 
            end_values = spopt.newton(self._compute_residual(initial_value,rhs_f,delta_t),initial_value,maxiter=100)

        return end_values
    
    @abstractmethod
    def _compute_residual(self,
                  initial_value: np.ndarray,
                  rhs_f: Callable[...,np.ndarray],
                  delta_t: float) -> np.ndarray:
        
        """
        construct the residual function residual(x), the roots of which will be computed numerically

        Parameters
        ----------
        initial_value: np.ndarray
            initial value w0
        rhs_f: function
            the right-hand side function describing the time evolution of w
        delta_t: float
            time step size
        
        Returns
        -------
        residual: function
            function of which the roots will be computed (residual(x) == 0, where x is the numerical solution of the implicit solver) 

        """
        
        pass

class Explicit(TimeIntegration):

    """
    This abstract class represents an explicit time integrator.

    ...

    Attributes
    ----------
    None

    Abstract methods from abstract class TimeIntegration
    -------------
    def integrate(self,initial_value,rhs_f,delta_t):
        integrates the equation dw/dt = rhs_f(w) with a time step of delta_t with an explicit time integration method

    Imherited methods from abstract class TimeIntegration
    -------------
    def __initi__(self):
        initializes the explicit time integrator 

    """

    @abstractmethod    
    def integrate(self,
                  initial_value: np.ndarray,
                  rhs_f: Callable[...,np.ndarray],
                  delta_t: float) -> np.ndarray:
        
        """
        integrates the equation dw/dt = rhs_f(w) starting from initial_value with a time step of delta_t

        Parameters
        ----------
        initial_value: np.ndarray
            initial value w0
        rhs_f: function
            the right-hand side function describing the time evolution of w
        delta_t: float
            time step size
        
        Returns
        -------
        end_values: np.ndarray
            final values 

        """
        
        pass

class Exact(TimeIntegration):

    """
    This class represents an exact time integrator.
    This integrator can be used when the analytical solution of the source term step is known.

    ...

    Attributes
    ----------
    None

    Implemented methods from abstract class TimeIntegration
    -------------
    def integrate(self,initial_value,rhs_f,delta_t):
        integrates the equation dw/dt = rhs_f(w) with a time step of delta_t

    """
   
    def integrate(self,
                  initial_value: np.ndarray,
                  rhs_f: Callable[...,np.ndarray],
                  delta_t: float) -> np.ndarray:
        
        """
        integrates the equation dw/dt = rhs_f.w (matrix-vector multiplication) exactly 
        starting from initial_value with a time step of delta_t.

        Parameters
        ----------
        initial_value: np.ndarray
            initial value w0
        rhs_f: function
            the right-hand side function describing the time evolution of w
        delta_t: float
            time step size
        
        Returns
        -------
        end_values: np.ndarray
            final values 

        """
        
        end_values = rhs_f(initial_value,delta_t)

        return end_values

class ImplicitEuler(Implicit):

    """
    This class represents an implicit Euler time integrator.

    ...

    Attributes
    ----------
    linear : boolean
        True if the source term can be written in linear form, false if the source term can not be written in linear form

    Implemented methods from abstract parent class Implicit
    -------------
    def _compute_residual(self,initial_value,rhs_f,delta_t):
        construct the residual function residual(x), the roots of which will be computed numerically

    """

    def _compute_residual(self,
                  initial_value: np.ndarray,
                  rhs_f: Callable[...,np.ndarray],
                  delta_t: float) -> np.ndarray:
        
        residual = lambda end_value : end_value - delta_t*rhs_f(end_value,delta_t) - initial_value

        return residual
    
class ExplicitEuler(Explicit):

    """
    This class represents an explicit Euler time integrator.

    ...

    Attributes
    ----------
    None

    Implemented methods from abstract parent class Explicit
    -------------
    def integrate(self,initial_value,rhs_f,delta_t):
        integrates the equation dw/dt = rhs_f(w) with a time step of delta_t

    """
    def integrate(self,
                  initial_value: np.ndarray,
                  rhs_f: Callable[...,np.ndarray],
                  delta_t: float) -> np.ndarray:
        
        """
        integrates the equation dw/dt = rhs_f(w) starting from initial_value with a time step of delta_t

        Parameters
        ----------
        initial_value: np.ndarray
            initial value w0
        rhs_f: function
            the right-hand side function describing the time evolution of w
        delta_t: float
            time step size
        
        Returns
        -------
        end_values: np.ndarray
            final values 

        """
        
        end_values = initial_value + delta_t*rhs_f(initial_value,delta_t)

        return end_values
