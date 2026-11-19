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
    Horton infiltration model.

    Horton's law gives a capacity f_p(t); the realised rate is the smaller of
    that capacity and the water supplied to the bed. The supply is the rain
    plus any water standing on the bed, which over one source step of length
    dt is

        I = min( f_p(t), R + h / dt )                  (ponded, default)

    so a dry bed (h = 0) infiltrates min(f_p, R), and a ponded bed with
    enough standing water infiltrates at full capacity f_p, even when f_p > R.
    The h / dt part also guarantees that one step never removes more water
    than the cell holds.

    The legacy closure of the thesis caps the rate by rainfall alone,

        I = min( f_p(t), R, h / dt )                   (limit_by_rainfall=True)

    under which standing water never feeds the ground. It is kept so that the
    thesis configurations reproduce their published figures.

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
    limit_by_rainfall : bool, optional
        If True, use the legacy rainfall-limited closure. Default False.
    """

    # Constructor
    def __init__(
            self,
            f0 : float,
            fc : float,
            k : float,
            eps : float = 1e-14,
            limit_by_rainfall : bool = False,
    ):
        self.f0 = f0
        self.fc = fc
        self.k = k
        self.eps = eps
        self.limit_by_rainfall = limit_by_rainfall

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

        Ponded (default):        I = min( f_p(t), R + h / dt )
        limit_by_rainfall=True:  I = min( f_p(t), R, h / dt )

        If h or dt is not given, the standing-water term is dropped and both
        closures reduce to min( f_p(t), R ).

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

        if not self.limit_by_rainfall:
            # Ponded closure: the supply is the rain plus the standing water
            supply = R
            if h is not None and dt is not None:
                h = np.asarray(h, dtype=np.float64)
                supply = R + np.maximum(h, 0.0) / max(float(dt), self.eps)
            I = np.maximum(np.minimum(fp, supply), 0.0)
            return np.float64(I) if I.ndim == 0 else I

        # Legacy rainfall-limited closure (thesis), unchanged
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
            f"k={self.k}, eps={self.eps}, "
            f"limit_by_rainfall={self.limit_by_rainfall})"
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
        If True, cap infiltration by the rainfall rate, and the availability
        bound below becomes h / dt (rainfall-limited variant).
    limit_by_available_water : bool, optional
        If True, cap infiltration by the water available over one source
        step, R + h / dt (thesis eq. 4.12): the water standing in the cell
        plus the rain supplied during the step. One explicit step then cannot
        produce a negative depth.

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

        For I0 > 0 (thesis eq. 4.12), with the default caps:
            I(t) = min( I0, R + h / dt )

        With limit_by_rainfall=True (rainfall-limited variant):
            I(t) = min( I0, R, h / dt )

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
            # `infiltrated_depth` may be called without a rainfall rate
            R = np.asarray(0.0 if rainfall is None else rainfall,
                           dtype=np.float64)

            # Apply the rainfall limit if enabled
            if self.limit_by_rainfall:
                I = np.minimum(I, R)

            # Availability bound over one source step. The supply is the
            # standing water plus, unless the rate is already capped by
            # rainfall, the rain of the step: R + h / dt (thesis eq. 4.12).
            # eps guards the division for dt = 0.
            if self.limit_by_available_water and h is not None and dt is not None:
                h = np.asarray(h, dtype=np.float64)
                available_rate = np.maximum(h, 0.0) / max(float(dt), self.eps)
                if not self.limit_by_rainfall:
                    available_rate = available_rate + np.maximum(R, 0.0)
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
