from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import pde
import mesh
import simulation

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

class SWME1DPlotAdaptive(Plotting):

    """
    This class represents a plotting object for the plotting of numerical results of the 1D SWME of an adaptive simulation.

    ...

    Attributes
    ----------
    pde_type : SWME1D
        the 1D SWME object
    mesh : RectangularMesh
        The simulation mesh
    simulation : SpatiallyAdaptiveSimulation1D
        The adaptive 1D simulation object

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
                 simulation: simulation.SpatiallyAdaptiveSimulation1D):
        """
        initializes the adaptive SWME1D plotting object

        Parameters
        ------------
        pde_type : SWME1D
            the SWME1D moment model
        mesh : RectangularMesh
            the numerical simulation mesh
        simulation : SpatiallyAdaptiveSimulation1D
            the adaptive 1D simulation object

        Returns
        --------        
        None

        """
        self.pde_type = pde_type 
        self.mesh = mesh
        self.simulation = simulation

    def plot(self,data_array):
        
        z = np.linspace(0,1,100)

        velocity_profile = self.pde_type.compute_vertical_velocity_profile(self.simulation.max_order,
                                                                    data_array,
                                                                    z)
        order = self.simulation.max_order

        print('total mass = ',np.sum(data_array[:,1]*data_array[:,2]))

        plt.figure()
        plt.subplot(4,4,1)
        plt.plot(velocity_profile[np.floor_divide(self.mesh.resolution,2),:], z)
        plt.title('Velocity profile')

        plt.subplot(4,4,2)
        plt.plot(self.mesh.cell_center_positions, data_array[:,1])
        plt.title('Height')

        plt.subplot(4,4,3)
        plt.plot(self.mesh.cell_center_positions, data_array[:,2])
        plt.title('Velocity')

        k = 4
        for i in range(order):
            plt.subplot(4,4,k)
            plt.plot(self.mesh.cell_center_positions, data_array[:,3+i])
            plt.title('alpha_'+str(i))
            k += 1

        plt.subplot(4,4,k)
        # plt.plot(self.mesh.cell_center_positions[:-1],height_gradient)
        plt.plot(self.mesh.cell_center_positions,self.simulation.breakdown_estimators[:,2])
        plt.title('Height gradient')

        plt.subplot(4,4,k+1)
        # plt.plot(self.mesh.cell_center_positions[:-1],momentum_gradient)
        plt.plot(self.mesh.cell_center_positions,self.simulation.breakdown_estimators[:,3])
        plt.title('Velocity gradient')

        plt.subplot(4,4,k+2)
        plt.plot(self.mesh.cell_center_positions,self.simulation.dom_decomp_val_res1)
        plt.title('domain_decomposition_values 1')

        plt.subplot(4,4,k+3)
        plt.plot(self.mesh.cell_center_positions,self.simulation.dom_decomp_val_res2)
        plt.title('domain_decomposition_values 2')

        plt.subplot(4,4,k+4)
        plt.plot(self.mesh.cell_center_positions,self.simulation.breakdown_estimators[:,2])
        plt.scatter(self.mesh.cell_center_positions,(data_array[:,-1]*np.max(self.simulation.breakdown_estimators[:,2])+(5-data_array[:,-1])*np.min(self.simulation.breakdown_estimators[:,2]))/5,s=5,color = 'hotpink')
        plt.title('orders vs height-gradient')

        plt.subplot(4,4,k+5)
        # plt.plot(self.mesh.cell_center_positions[:-1],_pde.compute_breakdown_criterion(data_array[:,1:],orders,number_of_variables,'last_moment',self.mesh.resolution-1,delta_x))
        plt.plot(self.mesh.cell_center_positions,self.simulation.breakdown_estimators[:,1])
        plt.title('Absolute value last moment')

        plt.subplot(4,4,k+6)
        # plt.plot(self.mesh.cell_center_positions[:-1],_pde.compute_breakdown_criterion(data_array[:,1:],orders,number_of_variables,'source_term',self.mesh.resolution-1,delta_x))
        plt.plot(self.mesh.cell_center_positions,self.simulation.breakdown_estimators[:,0])
        plt.title('source term')

        plt.show()

class HME1DPlotClassical(Plotting):

    """
    This class represents a plotting object for the plotting of numerical results of the 1D HME of a classical simulation.

    ...

    Attributes
    ----------
    pde_type : HME1D
        the 1D HME object
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
                 pde_type: pde.HermiteMomentEquations,
                 mesh: mesh.RectangularMesh,
                 simulation: simulation.ClassicalSimulation1D):
        """
        initializes the classical HME1D plotting object

        Parameters
        ------------
        pde_type : HME1D
            the HME1D moment model
        mesh : RectangularMesh
            the numerical simulation mesh
        simulation : ClassicalSimulation1D
            the classical 1D simulation object

        Returns
        --------        
        None

        """
        self.pde_type = pde_type 
        self.mesh = mesh
        self.simulation = simulation

    def plot(self,data_array):
        
        order = self.simulation.order

        plt.figure()

        plt.subplot(4,4,1)
        plt.plot(self.mesh.cell_center_positions, data_array[:,1])
        plt.title('Density')

        plt.subplot(4,4,2)
        plt.plot(self.mesh.cell_center_positions, data_array[:,2])
        plt.title('Velocity')

        plt.subplot(4,4,3)
        plt.plot(self.mesh.cell_center_positions, data_array[:,3])
        plt.title('Temperature')

        k = 4
        for i in range(3,order+1):
            plt.subplot(4,4,k)
            plt.plot(self.mesh.cell_center_positions, data_array[:,i+1])
            plt.title('f_'+str(i))
            k += 1

        plt.show()

class HME1DPlotAdaptive(Plotting):

    """
    This class represents a plotting object for the plotting of numerical results of the 1D HME of an adaptive simulation.

    ...

    Attributes
    ----------
    pde_type : HME1D
        the 1D HME object
    mesh : RectangularMesh
        The simulation mesh
    simulation : SpatiallyAdaptiveSimulation1D
        The adaptive 1D simulation object

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
                 pde_type: pde.HermiteMomentEquations,
                 mesh: mesh.RectangularMesh,
                 simulation: simulation.SpatiallyAdaptiveSimulation1D):
        """
        initializes the adaptive HME1D plotting object

        Parameters
        ------------
        pde_type : HME1D
            the HME1D moment model
        mesh : RectangularMesh
            the numerical simulation mesh
        simulation : SpatiallyAdaptiveSimulation1D
            the adaptive 1D simulation object

        Returns
        --------        
        None

        """
        self.pde_type = pde_type 
        self.mesh = mesh
        self.simulation = simulation

    def plot(self,data_array):
        
        """
        Creates a plot

        Parameters
        ----------
        
        Returns
        -------

        """
        order = self.simulation.max_order

        plt.figure()

        plt.subplot(4,4,1)
        plt.plot(self.mesh.cell_center_positions, data_array[:,1])
        plt.title('Density')

        plt.subplot(4,4,2)
        plt.plot(self.mesh.cell_center_positions, data_array[:,2])
        plt.title('Velocity')

        plt.subplot(4,4,3)
        plt.plot(self.mesh.cell_center_positions, data_array[:,3])
        plt.title('Temperature')

        k = 4
        for i in range(3,order+1):
            plt.subplot(4,4,k)
            plt.plot(self.mesh.cell_center_positions, data_array[:,i+1])
            plt.title('f_'+str(i))
            k += 1

        plt.subplot(4,4,k)
        plt.plot(self.mesh.cell_center_positions, self.simulation.breakdown_estimators[:,0])
        plt.title('Absolute value last moment')

        plt.subplot(4,4,k+1)
        plt.plot(self.mesh.cell_center_positions, self.simulation.breakdown_estimators[:,1])
        plt.title('Density gradient')

        plt.subplot(4,4,k+2)
        plt.plot(self.mesh.cell_center_positions,data_array[:,0])
        plt.scatter(self.mesh.cell_center_positions,(data_array[:,-1]*np.max(data_array[:,0])+(5-data_array[:,-1])*np.min(data_array[:,0]))/8,s=8,color = 'hotpink')
        plt.title('orders vs density')

        plt.show()
