from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from . import pde
from . import mesh
from . import simulation

class Plotting(ABC):

    """
    This abstract class represents a plotting object (for the plotting of the simulation results).

    ...

    Attributes
    ----------
    pde_type : PDE
        The partial differential equation that has been simulated
    mesh : RectangularMesh
        The simulation mesh
    simulation : Simulation
        The simulation object
    
    Class methods
    -------------
    def __init__(self,pde_type):
        initializes the plotting object

    Abstract methods
    ---------------
    def plot(self):
        creates a plotting object and plots the simulation results
    """
    def __init__(self,
                 pde_type: pde.PDE,
                 mesh: mesh.RectangularMesh,
                 simulation: simulation.Simulation):

        """
        initializes the plotting object

        Parameters
        ------------
        pde_type : PDE
            the PDE model
        mesh : RectangularMesh
            the numerical simulation mesh
        simulation : Simulation
            the simulation object

        Returns
        --------        
        None

        """

        self.pde_type = pde_type 
        self.mesh = mesh
        self.simulation = simulation 

    @abstractmethod
    def plot(self,
             data_array: np.ndarray):
        """
        Creates a plot of the data listed in data_array

        Parameters
        ----------
        data_array : numpy array

        Returns
        -------
        None

        """
        pass

class SWME1DPlotClassical(Plotting):

    """
    This class represents a plotting object for the plotting of numerical results of the 1D SWME of a classical simulation.

    ...

    Attributes
    ----------
    pde_type : SWME1D
        the 1D SWME object
    mesh : RectangularMesh
        The simulation mesh
    simulation : ClassicalSimulation1D
        The classical 1D simulation object

    Implemented methods from abstract parent class 'Plotting'
    ---------------------------------------------------------
    def plot(self):
        creates a plotting object and plots the simulation results

    Methods overriden from abstract parent class 'Plotting
    ------------------------------------------------------
    def __init__(self,pde_type):
        initializes the plotting object

    """

    def __init__(self,
                 pde_type: pde.SWME1D,
                 mesh: mesh.RectangularMesh,
                 simulation: simulation.ClassicalSimulation1D):
        """
        initializes the classical SWME1D plotting object

        Parameters
        ------------
        pde_type : SWME1D
            the SWME1D moment model
        mesh : RectangularMesh
            the numerical simulation mesh
        simulation : ClassicalSimulation1D
            the simulation object

        Returns
        --------        
        None

        """

        self.pde_type = pde_type 
        self.mesh = mesh
        self.simulation = simulation

    def plot(self,
             data_array: np.ndarray):

        z = np.linspace(0,1,100)

        velocity_profile = self.pde_type.compute_vertical_velocity_profile(self.simulation.order,
                                                                    data_array,
                                                                    z)
        order = self.simulation.order

        print('total mass = ',np.sum(data_array[:,1]*data_array[:,2]))

        plt.figure()
        plt.subplot(3,3,1)
        plt.plot(velocity_profile[np.floor_divide(self.mesh.resolution,2),:], z)
        plt.title('Velocity profile')

        plt.subplot(3,3,2)
        if getattr(self.mesh, 'has_topography', False):
            # Over a non-flat bed the water depth h on its own is close to
            # unreadable - a lake at rest looks like an inverted bump. Show
            # the free surface h + Z against the bed instead, which is what
            # the well-balancing property is actually about.
            bed = self.mesh.bed_elevation[1:self.mesh.resolution+1]
            plt.plot(self.mesh.cell_center_positions, data_array[:,1] + bed,
                     label='h + Z')
            plt.plot(self.mesh.cell_center_positions, bed, 'k-', label='Z')
            plt.legend(fontsize='small')
            plt.title('Free surface over bed')
        else:
            plt.plot(self.mesh.cell_center_positions, data_array[:,1])
            plt.title('Height')

        plt.subplot(3,3,3)
        plt.plot(self.mesh.cell_center_positions, data_array[:,2])
        plt.title('Velocity')

        k = 4
        for i in range(order):
            plt.subplot(3,3,k)
            plt.plot(self.mesh.cell_center_positions, data_array[:,3+i])
            plt.title('alpha_'+str(i+1))
            k += 1

        plt.show()

