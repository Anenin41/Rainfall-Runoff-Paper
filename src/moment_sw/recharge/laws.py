#  Packages
import numpy as np

class AdmissibleMixingFriction:
    """
    Baseline admissible mixing-friction law. The mixing-induced friction that
    is implemented here is a direct result from the control-volume momentum
    balance approach of the report. Final result:

        f_R = alpha_R * R
        f_I = alpha_I * max(0, -I)

    Parameters
    ----------
    alpha_R : float
        Surface mixing-friction slope for rainfall.
    alpha_I : float
        Bed mixing-friction slope for exfiltration.

    Notes
    -----
    This is the minimal physically admissible closure as described by a control-
    volume momentum balance argument. It is not a fully derived microscopic law,
    and the closure changes per application.
    """

    # Constructor
    def __init__(self, alpha_R : float, alpha_I : float):
        if alpha_R < 0.0 or alpha_I < 0.0:
            raise ValueError("Mixing-induced slopes must be non-negative.")
        # alpha_R, alpha_I are dimensionless slopes
        self.alpha_R = alpha_R
        self.alpha_I = alpha_I

    # Evaluate f_R and f_I
    # f_I is only active for I < 0, i.e. water returning back into the domain 
    # from the ground (exfiltration).
    def evaluate(self, R, I, context = None, values = None):
        R = float(np.maximum(R, 0.0))
        I_minus = float(np.maximum(-I, 0.0))

        f_R = self.alpha_R * R
        f_I = self.alpha_I *  I_minus

        return f_R, f_I
    
    # String representation
    def __repr__(self):
        return (
            f"AdmissibleMixingFriction("
            f"alpha_R={self.alpha_R}, alpha_I={self.alpha_I})"
        )
    
class HortonInfiltration(object):
    """
    Simple Horton infiltration model for the first recharge test case.

    Parameters
    ----------
    f0 : float
        Initial infiltration capacity.
    fc : float
        Final infiltration capacity.
    k : float
        Exponential decay coefficient.
    eps : float, optional
        Small positive number to avoid division by zero.
    """

    # Constructor
    def __init__(self, f0 : float, fc : float, k : float, eps : float = 1e-14):
        self.f0 = f0
        self.fc = fc
        self.k = k
        self.eps = eps

        if self.f0 < 0.0 or self.fc < 0.0 or self.k < 0.0:
            raise ValueError("Horton parameters must be non-negative.")
        if self.f0 < self.fc:
            raise ValueError("Expected f0 >= fc for classical Horton model.")
        
    def capacity(self, t):
        """
        Horton infiltration capacity (exponential decay law):
            f(t) = fc + (f0 - fc) * exp(-k * t)

        Parameters
        ----------
        t : float or np.ndarray
            Time value(s). 

        Returns
        -------
        float or np.ndarray
            Infiltration capacity at time t.
        """
        t = np.asarray(t, dtype=np.float64)
        fp = self.fc + (self.f0 - self.fc) * np.exp(-self.k * t)
        return np.float64(fp) if fp.ndim == 0 else fp
    
    def rate(self, t, rainfall, h=None, dt=None):
        """
        Actual infiltration rate used in the recharge source term.
        
        Simplest Choice: I(t) = min( f_p(t), rainfall, h / dt )

        Parameters
        ----------
        t : float or np.ndarray
            Current time.
        rainfall : float or np.ndarray
            Rainfall rate R.
        h : float or np.ndarray, optional  
            Local water depth.
        dt : float, optional
            Current timestep.
        
        Returns
        -------
        float or np.ndarray
            Actual infiltration rate at time t.
        """
        fp = np.asarray(self.capacity(t), dtype=np.float64)
        R = np.asarray(rainfall, dtype=np.float64)

        I = np.minimum(fp, R)

        if h is not None and dt is not None:
            h = np.asarray(h, dtype=np.float64)

            # Dry cell treatment via tolerance eps to avoid division by zero
            available_rate = np.maximum(h, 0.0) / max(float(dt), self.eps)
            
            I = np.minimum(I, available_rate)

        I = np.maximum(I, 0.0)
        return np.float64(I) if I.ndim == 0 else I
    
    def infiltrated_depth(self, t, rainfall, dt, h=None):
        """
        Water depth removed by infiltration over one timestep.

        Parameters
        ----------
        t : float or np.ndarray
            Current time.
        rainfall : float or np.ndarray
            Rainfall rate R.
        dt : float
            Current timestep.
        h : float or np.ndarray, optional
            Local water depth.

        Returns
        -------
        float or np.ndarray
            Infiltrated water depth over the timestep.
        """
        I = self.rate(t = t, rainfall=rainfall, h=h, dt=dt)
        depth = np.asarray(I, dtype=np.float64) * float(dt)
        return np.float64(depth) if depth.ndim == 0 else depth
    
    # String representation
    def __repr__(self):
        return (
            f"HortonInfiltration(f0={self.f0}, fc={self.fc},  "
            f"k={self.k}, eps={self.eps})"
        )
    
class ConstantInfiltration(object):
    """
    Constant infiltration model for testing and debugging purposes.
    
    Parameters
    ----------
    I0 : float
        Constant infiltration rate.
    eps : float, optional
        Small positive number to avoid division by zero.
    limit_by_rainfall : bool, optional
        If True, cap infiltration by the rainfall rate.
    limit_by_available_water : bool, optional
        If True, cap infiltration by h / dt so that more water is not removed
        than is available in the cell over one timestep.

    Notes
    -----
    This class follows a signed convention, followed more analytically in the
    report. In essence:
        I0 > 0  :   infiltration (water leaves the surface flow)
        I0 < 0  :   exfiltration (water enters the surface flow from the bed)
    """

    # Constructor
    def __init__(
            self,
            I0 : float,
            eps : float = 1e-14,
            limit_by_rainfall : bool = False,
            limit_by_available_water : bool = True,
    ):
        self.I0 = I0
        self.eps = eps
        self.limit_by_rainfall = limit_by_rainfall
        self.limit_by_available_water = limit_by_available_water
        
    def capacity(self, t):
        """
        Constant infiltration capacity/rate.

        Parameters
        ----------
        t : float or np.ndarray
            Time value(s). Ignored in this constant infiltration model. Included
            only for compatibility with the solver interface.

        Returns
        -------
        float or np.ndarray
            Constant infiltration value.
        """

        t = np.asarray(t, dtype=np.float64)
        I = np.full_like(t, fill_value=self.I0, dtype=np.float64)
        return np.float64(I) if I.ndim == 0 else I
    
    def rate(self, t, rainfall, h = None, dt = None):
        """
        Actual infiltration rate used in the recharge source term. It follows
        a signed convention.

        Convention
        ----------
        I > 0   :   infiltration (water leaves the surface flow)
        I < 0   :   exfiltration (water enters the surface flow from the bed)

        For I0 > 0, optional rainfall / available-water caps are applied
        For I0 < 0, the value is returned unchanged

        Default testing behaviour:
            I(t) = I0

        Optional caps can be enabled:
            I(t) = min( I0, rainfall, h / dt )

        Parameters
        ----------
        t : float or np.ndarray
            Current time. Ignored in this constant infiltration model. Included
            only for compatibility with the solver interface.
        rainfall : float or np.ndarray
            Rainfall rate R.
        h : float or np.ndarray, optional
            Local water depth.
        dt : float, optional
            Current timestep.

        Returns
        -------
        float or np.ndarray
            Actual infiltration rate at time t.
        """
        I = np.asarray(self.capacity(t), dtype=np.float64)

        # Infiltration branch
        if np.all(I >= 0.0):
            # Apply the rainfall limit if enabled
                if self.limit_by_rainfall:
                    R = np.asarray(rainfall, dtype=np.float64)
                    I = np.minimum(I, R)

                # Apply dry cell treatment via tolerance eps to avoid division by zero
                if self.limit_by_available_water and h is not None and dt is not None:
                    h = np.asarray(h, dtype=np.float64)
                    available_rate = np.maximum(h, 0.0) / max(float(dt), self.eps)
                    I = np.minimum(I, available_rate)

                I = np.maximum(I, 0.0)
        
        # Total output and exfiltration branch
        return np.float64(I) if I.ndim == 0 else I
    
    def infiltrated_depth(self, t, rainfall = None, dt = 0.0, h = None):
        """
        Water depth removed by infiltration over one timestep.

        Parameters
        ----------
        t : float or np.ndarray
            Current time.
        rainfall : float or np.ndarray, optional
            Rainfall rate R.
        dt : float
            Current timestep.
        h : float or np.ndarray, optional
            Local water depth.

        Returns
        -------
        float or np.ndarray
            Infiltrated water depth over the timestep.
        """
        I = self.rate(t = t, rainfall=rainfall, h=h, dt=dt)
        depth = np.asarray(I, dtype=np.float64) * float(dt)
        return np.float64(depth) if depth.ndim == 0 else depth

    # String representation
    def __repr__(self):
        return (
            f"ConstantInfiltration(I0={self.I0}, eps={self.eps}, "
            f"limit_by_rainfall={self.limit_by_rainfall}, "
            f"limit_by_available_water={self.limit_by_available_water})"
        )

# Test Case & Sanity Check for the Horton Infiltration Model
def horton_test1():

    print("=== Horton Infiltration Sanity Check ===")
    print("Reference: Fernandez-Pato et al. (2016), Case 0")

    # Horton parameters as used in the paper (Case 0 / sandy soil)
    horton = HortonInfiltration(
        f0 = 1.977e-4,
        fc = 3.272e-5,
        k = 2.43e-3,
    )

    # Test 1: ponded soil & unlimited water check
    R_ponded = 1.0e-3   # larger than f0, so rainfall doesn't limit infiltration
    h_large = 1.0       # 1 meter of water, effectively unlimited for this test
    dt = 1.0            # 1 second, h/dt = 1.0 m/s >> infiltration capacity

    times_min = np.array([0, 1, 2, 5, 10, 20, 30, 60, 120, 180, 300],
                         dtype = float)
    times_sec = 60.0 * times_min

    print("PONDED / UNLIMITED-WATER TEST")
    print("Expect: I(t) = f_p(t)")
    print(f"{'t [min]':>8} {'f_p(t) [m/s]':>16} {'I(t) [m/s]':>16} {'F(t) [m]':>16}")

    for t_min, t_sec in zip(times_min, times_sec):
        fp = horton.capacity(t_sec)
        I = horton.rate(t = t_sec, rainfall = R_ponded, h = h_large, dt = dt)

        # Analytic cumulative infiltration corresponding to Horton capacity:
        # F(t) = fc*t + (f0 - fc)/k * (1 - exp(-k*t))
        F = (
            horton.fc * t_sec
            + (horton.f0 - horton.fc) / horton.k * (
                1.0 - np.exp(-horton.k * t_sec)
            )
        )

        print(f"{t_min:8.1f} {fp:16.6e} {I:16.6e} {F:16.6e}")

    print("Check: under ponded conditions, I(t) should match f_p(t) row by row")
    print()

    # Test 2: Rainfall limited checl
    R_limited = 1.0e-4

    print("RAINFALL-LIMITED TEST")
    print(f"Using R = {R_limited:.6e} m/s")
    print(f"{'t [min]':>8} {'f_p(t) [m/s]':>16} {'I(t) [m/s]':>16}")

    for t_min, t_sec in zip(times_min, times_sec):
        fp = horton.capacity(t_sec)
        I = horton.rate(t = t_sec, rainfall = R_limited, h = h_large, dt = dt)
        print(f"{t_min:8.1f} {fp:16.6e} {I:16.6e}")

    print("Check: early on, I(t) should equal rainfall (1.0e-4 m/s),")
    print("and later switch to the Horton capacity once f_p(t) drops below rainfall.")
    print()

    # Test 3: Available-water-limited check
    h_small = 1.0e-5
    dt_large = 1.0
    available_rate = h_small / dt_large

    I_avail = horton.rate(t=0.0, rainfall=R_ponded, h=h_small, dt=dt_large)

    print("AVAILABLE-WATER-LIMITED TEST")
    print(f"h = {h_small:.6e} m, dt = {dt_large:.6e} s")
    print(f"h/dt = {available_rate:.6e} m/s")
    print(f"I(0) = {I_avail:.6e} m/s")
    

    print("Check: I(0) should equal h/dt in this test.")
    print()

    # Test 4: Constant infiltration model check
    constant_model = ConstantInfiltration(
        I0=0.05,
        limit_by_rainfall=False,
        limit_by_available_water=True,
    )

    I_const = constant_model.rate(t=0.0, rainfall=R_ponded, h=1.0e-3, dt=1.0e-2)

    print("=== Constant infiltration sanity check ===")
    print("Constant Infiltration Model:", constant_model)
    print(f"Actual Infiltration I(t) = {I_const:.6e}")

def horton_test2():

    import matplotlib.pyplot as plt

    print()
    print("=== Horton Infiltration Sanity Check ===")
    print("Reference: Fernandez-Pato et al. (2016), Case 0")

    horton = HortonInfiltration(
        f0 = 1.977e-4,
        fc = 3.272e-5,
        k = 2.43e-3,
    )

    # Time axis: 0 to 300 minutes
    t_minutes = np.linspace(0.0, 300.0, 1200)
    t_seconds = 60.0 * t_minutes

    # Ponded soil / effectively unlimited water
    R_ponded = 1.0e-3
    h_large = 1.0
    dt = 1.0

    fp = horton.capacity(t_seconds)
    I_horton = horton.rate(
        t = t_seconds,
        rainfall = R_ponded,
        h = h_large,
        dt = dt,
    )

    # Cumulative infiltration by trapezoidal rule
    F_horton = np.zeros_like(t_seconds)
    F_horton[1:] = np.cumsum(
        0.5 * (I_horton[1:] + I_horton[:-1]) * np.diff(t_seconds)
    )

    # Plot 1: capacity and actual infiltration
    fig, ax = plt.subplots(figsize = (8, 5))
    ax.plot(t_minutes, fp, label="Horton Capacity f_p")
    ax.plot(t_minutes, I_horton, label="Actual infiltration I", linestyle="--")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Rate [m/s]")
    ax.set_title("Horton infiltration law: rate test")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    
    # Plot 2: cumulative infiltrated depth
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_minutes, F_horton, label="Cumulative infiltration")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Infiltrated depth [m]")
    ax.set_title("Horton infiltration law: cumulative infiltration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    # Rainfall limited Horton test case
    R_small = 5.0e-5
    I_rain_limited = horton.rate(
        t = t_seconds,
        rainfall = R_small,
        h = h_large,
        dt = dt,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_minutes, horton.capacity(t_seconds), label="Capacity f_p")
    ax.plot(t_minutes, np.full_like(t_minutes, R_small), label="Rainfall R", linestyle=":")
    ax.plot(t_minutes, I_rain_limited, label="Actual infiltration I", linestyle="--")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Rate [m/s]")
    ax.set_title("Horton law: rainfall-limited test")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()

def horton_test_from_config(config_path="../config/config.txt"):
    import configparser
    import os
    import matplotlib.pyplot as plt
    plt.style.use('tableau-colorblind10')

    config = configparser.ConfigParser()
    config.read(config_path)

    pde_info = config["pde_information"]
    num_info = config["numerical_method_information"]

    R = pde_info.getfloat("rainfall_rate")
    f0 = pde_info.getfloat("horton_f0")
    fc = pde_info.getfloat("horton_fc")
    k = pde_info.getfloat("horton_k")
    t_end = num_info.getfloat("t_end")

    horton = HortonInfiltration(f0=f0, fc=fc, k=k)

    t = np.linspace(0.0, t_end, 1200)

    # Use the same idealized unlimited-water setting as the built-in test.
    h_large = 1.0
    dt = 1.0

    fp = horton.capacity(t)
    I = horton.rate(t=t, rainfall=R, h=h_large, dt=dt)

    net_source = R - I

    # Expected height response for a spatially uniform, initially still test.
    h = np.zeros_like(t)
    h[0] = 1.0
    h[1:] = h[0] + np.cumsum(
        0.5 * (net_source[1:] + net_source[:-1]) * np.diff(t)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw the two limiting curves first and make them slightly wider.
    # This leaves a visible coloured outline below the green curve.
    ax.plot(t, fp, label=r"Horton capacity $f_p(t)$",
            color="#0072B2", linewidth=3.0, linestyle="-", zorder=1)
    ax.plot(t, np.full_like(t, R), label=r"Rainfall rate $R$", 
            color="#E69F00", linewidth=2.0, linestyle="-.", zorder=2)
    ax.plot(t, I, label=r"Actual infiltration $I(t)$", 
            color="#000000", linewidth=1.5, linestyle="--", zorder=3)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("InfiltrationRate [m/s]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, net_source, label=r"Net source $R-I(t)$", linewidth=2.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Net Source Rate [m/s]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, h, label=r"Mean", linewidth=2.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$h$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    horton_test1()
    horton_test2()
    horton_test_from_config()