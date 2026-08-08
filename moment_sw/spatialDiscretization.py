from abc import ABC, abstractmethod
import numpy as np
from collections.abc import Callable
import mpmath as mp

class SpatialDiscretization(ABC):

    """
    This abstract class represents a spatial discretization.

    ...

    Attributes
    ----------
    None

    
    Abstract methods
    ----------------
    def compute_fluctuation(self):
        computes a fluctuation between two cells


    Instance methods
    -----------------
    def __init__(self):
        initializes the object, but does not do anything


    """

    def __init__(self):
        pass

    @abstractmethod
    def compute_fluctuation(self):

        """
        Documented in the child classes

        """
        pass

class PVM(SpatialDiscretization,ABC):

    """
    This abstract class represents a polynomial viscosity method.

    ...

    Attributes
    ----------
    None

    Methods implemented from the abstract class SpatialDiscretization
    -----------------------------------------------------------------
    def compute_fluctuation(self,value_left,value_right,system_matrix,direction,delta_t,delta_x):
        computes the fluctuation between two cells with values value_left and value_right
    
    Abstract methods    
    -----------------
    def compute_viscosity(self,roe_matrix,delta_t,delta_x):
        computes the numerical viscosity matrix


    Instance methods
    -------------
    def compute_generalized_roe_and_viscosity(self,value_left,value_right,system_matrix,direction,delta_t,delta_x):
        compute the generalized roe matrix and the viscosity matrix between two cells with values value_left and value_right
            

    """

    def compute_fluctuation(self,
                            value_left: np.ndarray,
                            value_right: np.ndarray,
                            system_matrix: Callable[...,np.ndarray],
                            delta_t: float,
                            delta_x: float) -> tuple[np.ndarray,np.ndarray]:
        
        """
        Computes the fluctuations between two cells containing the values value_left and value_right

        Parameters
        ----------
        value_left: np.ndarray
            value of the cell left of the boundary
        value_right: np.ndarray
            value of the cell right of the boundary
        system_matrix: function
            function that computes the system matrix along the path between value_left and value_right
        delta_t: float
            time step size
        delta_x: float
            spatial discretization step size
        
        Returns
        -------
        fluctuation_min: np.ndarray
            fluctuation between two cells containing the values value_left and value_right in negative direction
        fluctuation_plus: np.ndarray
            fluctuation between two cells containing the values value_left and value_right in positive direction

        """
        # Nodes on [0, 1]
        quadrature_nodes = [
            (1 - (1/3) * np.sqrt((5 + 2*np.sqrt(10/7))/3)) / 2,
            (1 - (1/3) * np.sqrt((5 - 2*np.sqrt(10/7))/3)) / 2,
            1/2,
            (1 + (1/3) * np.sqrt((5 - 2*np.sqrt(10/7))/3)) / 2,
            (1 + (1/3) * np.sqrt((5 + 2*np.sqrt(10/7))/3)) / 2
        ]

        # Weights on [0, 1]
        quadrature_weights = [
            (322 - 13*np.sqrt(70)) / 1800,
            (322 + 13*np.sqrt(70)) / 1800,
            128 / 450,
            (322 + 13*np.sqrt(70)) / 1800,
            (322 - 13*np.sqrt(70)) / 1800
        ]

        # quadrature_nodes = [1/2]
        # quadrature_weights = [1]

        generalized_roe = 0
        for i in range(len(quadrature_nodes)):
            generalized_roe += quadrature_weights[i]*(system_matrix((1-quadrature_nodes[i])*value_left+(quadrature_nodes[i])*value_right))
        viscosity = np.dot(self.compute_viscosity(generalized_roe,delta_t,delta_x),value_right-value_left)
        generalized_roe = np.dot(generalized_roe,value_right-value_left)
        fluctuation_min = (generalized_roe - viscosity)/2
        fluctuation_plus = (generalized_roe + viscosity)/2

        return fluctuation_min, fluctuation_plus
    
    def compute_generalized_roe_and_viscosity(self,
                                              value_left,
                                              value_right,
                                              system_matrix,
                                              delta_t,
                                              delta_x):

        """
        Computes the generalized roe matrix and the viscosity matrix between two cells with values value_left and value_right.
        The fluctuations in the PVM scheme can be written in the form D^+- = A^+- . (value_right - value_left).
        This function returns the matrices A^+ and A^-.

        Parameters
        ----------
        value_left: np.ndarray
            value of the cell left of the boundary
        value_right: np.ndarray
            value of the cell right of the boundary
        system_matrix: function
            function that computes the system matrix along the path between value_left and value_right
        delta_t: float
            time step size
        delta_x: float
            spatial discretization step size
        
        Returns
        -------
        fluct_matrix_min: np.ndarray
            fluctuation matrix in the negative direction
        fluct_matrix_plus: np.ndarray
            fluctuation matrix in the positive direction

        """

        # Nodes on [0, 1]
        quadrature_nodes = [
            (1 - (1/3) * np.sqrt((5 + 2*np.sqrt(10/7))/3)) / 2,
            (1 - (1/3) * np.sqrt((5 - 2*np.sqrt(10/7))/3)) / 2,
            1/2,
            (1 + (1/3) * np.sqrt((5 - 2*np.sqrt(10/7))/3)) / 2,
            (1 + (1/3) * np.sqrt((5 + 2*np.sqrt(10/7))/3)) / 2
        ]

        # Weights on [0, 1]
        quadrature_weights = [
            (322 - 13*np.sqrt(70)) / 1800,
            (322 + 13*np.sqrt(70)) / 1800,
            128 / 450,
            (322 + 13*np.sqrt(70)) / 1800,
            (322 - 13*np.sqrt(70)) / 1800
        ]

        # quadrature_nodes = [1/2]
        # quadrature_weights = [1]

        generalized_roe = 0
        for i in range(len(quadrature_nodes)):
            generalized_roe += quadrature_weights[i]*(system_matrix((1-quadrature_nodes[i])*value_left+(quadrature_nodes[i])*value_right))
        viscosity = self.compute_viscosity(generalized_roe,delta_t,delta_x)
        fluct_matrix_min = (generalized_roe - viscosity)/2
        fluct_matrix_plus = (generalized_roe + viscosity)/2

        return fluct_matrix_min, fluct_matrix_plus

    @abstractmethod
    def compute_viscosity(self,
                          roe_matrix: np.ndarray,
                          delta_t: float,
                          delta_x: float):
        
        """
        Computes the numerical viscosity matrix

        Parameters
        ----------
        roe_matrix: np.ndarray
            roe matrix at the interface
        delta_t: float
            time step size
        delta_x: float
            spatial discretization step size
        
        Returns
        -------
        viscosity: np.ndarray
            numerical viscosity matrix

        """

        pass

class PRICE(PVM):
    """
    This class represents the PRICE scheme, a PVM scheme with viscosity function Q(A) = delta_x/(2*delta_t)*I+delta_t/(2*delta_x)*A^2.
    This method is the arithmetic average of the Lax-Friedrichs method and the Lax-Wendroff method. 

    ...

    Attributes
    ----------
    None

    
    Methods inherited from abtract parent class PVM
    ------------------------------------------------
    def compute_fluctuation(self,value_left,value_right,system_matrix,direction,delta_t,delta_x):
        computes the fluctuation between two cells with values value_left and value_right
    def compute_generalized_roe_and_viscosity(self,value_left,value_right,system_matrix,direction,delta_t,delta_x):
        compute the generalized roe matrix and the viscosity matrix between two cells with values value_left and value_right

    Methods implemented from abstract parent class PVM
    def compute_viscosity(self,roe_matrix,delta_t,delta_x):
        computes the viscosity matrix for the PRICE scheme

    """

    def compute_viscosity(self,
                          roe_matrix: np.ndarray,
                          delta_t: float,
                          delta_x: float):
        
        viscosity = 0.5*delta_x/delta_t*np.identity(np.shape(roe_matrix)[0])+0.5*delta_t/delta_x*roe_matrix@roe_matrix 
        return viscosity
    
class LF(PVM):
    """
    This class represents the Lax-Friedrichs scheme, a PVM scheme with viscosity function Q(A) = delta_x/delta_t*I. 

    ...

    Attributes
    ----------
    None

    
    Methods inherited from abtract parent class PVM
    ------------------------------------------------
    def compute_fluctuation(self,value_left,value_right,system_matrix,direction,delta_t,delta_x):
        computes the fluctuation between two cells with values value_left and value_right
    def compute_generalized_roe_and_viscosity(self,value_left,value_right,system_matrix,direction,delta_t,delta_x):
        compute the generalized roe matrix and the viscosity matrix between two cells with values value_left and value_right

    Methods implemented from abstract parent class PVM
    def compute_viscosity(self,roe_matrix,delta_t,delta_x):
        computes the viscosity matrix for the Lax-Friedrichs scheme

    """

    def compute_viscosity(self,
                          roe_matrix: np.ndarray,
                          delta_t: float,
                          delta_x: float):
        
        viscosity = delta_x/delta_t*np.identity(np.shape(roe_matrix)[0])
        return viscosity

class Roe(PVM):
    """
    This class represents the Roe scheme, a PVM scheme with viscosity function Q(A) = |A|. 

    ...

    Attributes
    ----------
    None

    
    Methods inherited from abtract parent class PVM
    ------------------------------------------------
    def compute_fluctuation(self,value_left,value_right,system_matrix,direction,delta_t,delta_x):
        computes the fluctuation between two cells with values value_left and value_right
    def compute_generalized_roe_and_viscosity(self,value_left,value_right,system_matrix,direction,delta_t,delta_x):
        compute the generalized roe matrix and the viscosity matrix between two cells with values value_left and value_right

    Methods implemented from abstract parent class PVM
    def compute_viscosity(self,roe_matrix,delta_t,delta_x):
        computes the viscosity matrix for the Roe scheme

    """

    def compute_viscosity(self,
                          roe_matrix: np.ndarray,
                          delta_t: float,
                          delta_x: float):

        # Eigen-decomposition: A = R D R^-1
        eigenvalues, R = np.linalg.eig(roe_matrix)
        
        # Construct |D|
        D_abs = np.diag(np.abs(eigenvalues))
        
        # Compute inverse of R
        R_inv = np.linalg.inv(R)
        
        # Return B = R |D| R^-1
        viscosity = R @ D_abs @ R_inv

        return viscosity
    
class Osher(PVM):
    """
    TODO

    """

    def compute_fluctuation(self,
                            value_left: np.ndarray,
                            value_right: np.ndarray,
                            system_matrix: Callable[...,np.ndarray],
                            delta_t: float,
                            delta_x: float) -> tuple[np.ndarray,np.ndarray]:
        
        # Nodes on [0, 1]
        quadrature_nodes = [
            (1 - (1/3) * np.sqrt((5 + 2*np.sqrt(10/7))/3)) / 2,
            (1 - (1/3) * np.sqrt((5 - 2*np.sqrt(10/7))/3)) / 2,
            1/2,
            (1 + (1/3) * np.sqrt((5 - 2*np.sqrt(10/7))/3)) / 2,
            (1 + (1/3) * np.sqrt((5 + 2*np.sqrt(10/7))/3)) / 2
        ]

        # Weights on [0, 1]
        quadrature_weights = [
            (322 - 13*np.sqrt(70)) / 1800,
            (322 + 13*np.sqrt(70)) / 1800,
            128 / 450,
            (322 + 13*np.sqrt(70)) / 1800,
            (322 - 13*np.sqrt(70)) / 1800
        ]

        # quadrature_nodes = [1/2]
        # quadrature_weights = [1]

        generalized_roe = 0
        viscosity = 0
        for i in range(len(quadrature_nodes)):
            generalized_roe_point = quadrature_weights[i]*(system_matrix((1-quadrature_nodes[i])*value_left+(quadrature_nodes[i])*value_right))
            generalized_roe += generalized_roe_point
            viscosity += self.compute_viscosity(generalized_roe_point,delta_t,delta_x)
        viscosity = np.dot(viscosity,value_right-value_left)
        generalized_roe = np.dot(generalized_roe,value_right-value_left)
        fluctuation_min = (generalized_roe - viscosity)/2
        fluctuation_plus = (generalized_roe + viscosity)/2

        return fluctuation_min, fluctuation_plus
    
    def compute_generalized_roe_and_viscosity(self,
                                              value_left,
                                              value_right,
                                              system_matrix,
                                              delta_t,
                                              delta_x):

        # Nodes on [0, 1]
        quadrature_nodes = [
            (1 - (1/3) * np.sqrt((5 + 2*np.sqrt(10/7))/3)) / 2,
            (1 - (1/3) * np.sqrt((5 - 2*np.sqrt(10/7))/3)) / 2,
            1/2,
            (1 + (1/3) * np.sqrt((5 - 2*np.sqrt(10/7))/3)) / 2,
            (1 + (1/3) * np.sqrt((5 + 2*np.sqrt(10/7))/3)) / 2
        ]

        # Weights on [0, 1]
        quadrature_weights = [
            (322 - 13*np.sqrt(70)) / 1800,
            (322 + 13*np.sqrt(70)) / 1800,
            128 / 450,
            (322 + 13*np.sqrt(70)) / 1800,
            (322 - 13*np.sqrt(70)) / 1800
        ]

        # quadrature_nodes = [1/2]
        # quadrature_weights = [1]

        generalized_roe = 0
        viscosity = 0
        for i in range(len(quadrature_nodes)):
            generalized_roe_point = quadrature_weights[i]*(system_matrix((1-quadrature_nodes[i])*value_left+(quadrature_nodes[i])*value_right))
            generalized_roe += generalized_roe_point
            viscosity += self.compute_viscosity(generalized_roe_point,delta_t,delta_x)
        fluct_matrix_min = (generalized_roe - viscosity)/2
        fluct_matrix_plus = (generalized_roe + viscosity)/2

        return fluct_matrix_min, fluct_matrix_plus


    def compute_viscosity(self,
                          roe_matrix: np.ndarray,
                          delta_t: float,
                          delta_x: float):

        # Eigen-decomposition: A = R D R^-1
        eigenvalues, R = np.linalg.eig(roe_matrix)
        
        # Construct |D|
        D_abs = np.diag(np.abs(eigenvalues))
        
        # Compute inverse of R
        R_inv = np.linalg.inv(R)
        
        # Return B = R |D| R^-1
        viscosity = R @ D_abs @ R_inv

        return viscosity