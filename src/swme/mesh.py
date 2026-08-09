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

        # Bottom topography Z(x), sampled per cell and including the two ghost
        # cells, so it is indexed exactly like the state array `values`:
        # index 0 = left ghost, 1..resolution = physical, resolution+1 = right
        # ghost. Zero (flat bed) until `set_bed_elevation` says otherwise, so
        # every config that never mentions topography is untouched.
        self.bed_elevation = np.zeros(self.resolution + 2, dtype=np.float64)
        self.bed_elevation_function = None
        self.bed_boundary_condition = None
        self.has_topography = False

    def _compute_cell_centers(self):
        cell_centers = np.linspace(self.boundaries[0], self.boundaries[1], self.resolution)
        return cell_centers

    def set_bed_elevation(self,
                          z_of_x,
                          boundary_condition: str = 'INFLOW_OUTFLOW') -> None:
        """Sample a bed-elevation profile Z(x) onto the grid.

        Parameters
        ----------
        z_of_x : callable
            Maps an array of positions to bed elevations; build one with
            `swme.topography.get_bed_profile`.
        boundary_condition : str
            Must match the simulation's own boundary condition, because the
            two ghost cells of Z have to be filled the same way the state's
            are, or the interface fluctuations at the domain edges see an
            inconsistent (U, Z) pair. 'PERIODIC' wraps; anything else is
            treated as zero-gradient extrapolation, matching
            `ClassicalSimulation1D._update_boundary_conditions`.

        Notes
        -----
        `has_topography` is set from whether the sampled bed is actually
        non-zero, not merely from this method having been called: a profile
        that evaluates to zero everywhere (e.g. `flat`) leaves the solver on
        the plain, non-augmented path and therefore reproduces pre-topography
        results bit for bit.
        """
        positions = np.asarray(self.cell_center_positions, dtype=np.float64)
        bed = np.zeros(self.resolution + 2, dtype=np.float64)
        bed[1:self.resolution + 1] = np.asarray(
            z_of_x(positions), dtype=np.float64
        )

        if boundary_condition == 'PERIODIC':
            bed[0] = bed[self.resolution]
            bed[self.resolution + 1] = bed[1]
        else:
            bed[0] = bed[1]
            bed[self.resolution + 1] = bed[self.resolution]

        if not np.isfinite(bed).all():
            raise ValueError(
                f"Bed elevation profile produced non-finite values: {bed}"
            )

        self.bed_elevation = bed
        self.bed_elevation_function = z_of_x
        # Recorded so the simulation can catch a mismatch against its own
        # boundary condition, which would otherwise be a silent wrong answer
        # confined to the two edge interfaces.
        self.bed_boundary_condition = boundary_condition
        self.has_topography = bool(np.any(bed != 0.0))


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
