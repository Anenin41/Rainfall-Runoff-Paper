from abc import ABC, abstractmethod
import numpy as np

#TODO: only rectangular, quadrilateral meshes with constant cell size are considered here. Extend to other types of mesh.
class RectangularMesh(ABC):

    """
    This interface represents a rectangular mesh.

    ...

    Attributes
    ----------
    boundaries : list of integers (if 1D) or list of list of integers (if 2D)
        the boundaries of the physical domain
    resolution : int (if 1D) or list of integers (if 2D)
        number of cells in each dimension

    
    Abstract methods
    -------
    def _compute_cell_centers():
        computes the cell centers of the mesh.
    """

    @abstractmethod
    def __init__(self, boundaries, resolution):
        """
        Constructs all the necessary attributes for the PDE object.

        Parameters
        ----------
        boundaries : list of integers (if 1D) or list of list of integers (if 2D)
            the boundaries of the physical domain
        resolution : int (if 1D) or list of integers (if 2D)
            number of cells in each dimension
        """
        
        self.boundaries = boundaries
        self.resolution = resolution

        self.cell_center_positions = self._compute_cell_centers()

    @abstractmethod
    def _compute_cell_centers(self):
        """
        Helper function that computes the cell centers of the cells in the mesh.

        Parameters
        ----------
        None
        
        
        Returns
        -------
        cell_centers: numpy 1D array
            cell centers of the cells

        """

        pass

class UniformRectangularMesh1D(RectangularMesh):

    """
    This class represents a uniform rectangular mesh in one dimension.

    ...

    Attributes
    ----------
    boundaries : list of integers 
        the boundaries of the physical domain
    resolution : int 
        number of cells 

    
    Implemented methods from interface Rectangular Mesh
    -------
    def _compute_cell_centers():
        computes the cell centers of the mesh.
    """

    def __init__(self, boundaries, resolution):
        self.boundaries = boundaries
        self.resolution = resolution

        self.cell_center_positions = self._compute_cell_centers()

    def _compute_cell_centers(self):
        cell_centers = np.linspace(self.boundaries[0], self.boundaries[1], self.resolution)
        return cell_centers
    
class UniformRectangularMesh2D(RectangularMesh):
    def __init__(self, boundaries, resolution):
        self.boundaries = boundaries
        self.resolution = resolution

        self.cell_center_positions = self._compute_cell_centers()

    def _compute_cell_centers(self):
        # cell_centers_x = np.linspace(self.boundaries[0,0], self.boundaries[0,1], self.resolution[0])
        cell_centers_x = np.linspace(self.boundaries[0,0], self.boundaries[0,1], self.resolution[0]+2)[1:-1]
        cell_centers_y = np.linspace(self.boundaries[1,0], self.boundaries[1,1], self.resolution[1]+2)[1:-1]
        cell_centers = [cell_centers_x,cell_centers_y]
        return cell_centers
