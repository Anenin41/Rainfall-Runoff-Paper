# Packages
import numpy as np

def _evaluate_rainfall_and_infiltration(
        values,
        rainfall,
        infiltration_model,
        context,
        expected_size : int,
        eps : float = 1e-14,
    ):
    """
    Shared helper function to evaluate rainfall and infiltration.
        - Validates state size,
        - Extracts h,
        - Evaluates rainfall R,
        - Evaluates infiltration I,
        - applies dry-cell treatment.
    """
    U = np.asarray(values, dtype = np.float64)
    if U.shape[0] != expected_size:
        raise ValueError(
            f"Expected state vector of length {expected_size}, got {U.shape[0]}."
        )
    
    # Extract height
    h = U[0]

    # Dry-cell treatment
    if h <= eps:
        return U, h, 0.0, 0.0, True
    
    # Evaluate rainfall R
    R = rainfall(context) if callable(rainfall) else np.float64(rainfall)

    # Evaluate infiltration I
    I = infiltration_model.rate(
        t = context.time,
        rainfall = R,
        h = h,
        dt = context.dt,
    )

    return U, h, np.float64(R), np.float64(I), False

# ============================================================================ #
# N = 0
# ============================================================================ #

def compute_recharge_source_n0(
      values,
      rainfall,
      infiltration_model,
      context,
      eps : float = 1e-14,  
    ):
    """
    Compute the pure rainfall-infiltration source term S_{R, I}(U) for the 
    zeroth order SWE system.

    State:
        U = [h, h * u_m]

    Source:
        S_{R, I}(U) = [
        R - I,
        (R - I) * u_m
        ]

    Parameters
    ----------
    values : array_like
        Conserved state variables [h, h * u_m].
    rainfall : float or callable
        Rainfall rate R. If callable, it is evaluated as rainfall(context).
    infiltration_model : object
        A customly encoded infiltration model. Must provide a method
            rate(t, rainfall, h, dt) or similar
    context : SourceContext
        Runtime source context.
    eps : float
        Small tolerance to avoid division by zero.

    Returns
    -------
    np.ndarray
        Recharge source vector of shape (2,).
    dict
        Diagnostics: R, I, h, u_m
    """
    # First evaluate rainfall and infiltration, with dry-cell treatment
    U, h, R, I, dry = _evaluate_rainfall_and_infiltration(
        values = values,
        rainfall = rainfall,
        infiltration_model = infiltration_model,
        context = context,
        expected_size = 2,
        eps = eps,
    )

    # If dry, return zero source and diagnostics with zeros
    if dry:
        return np.zeros(2, dtype = np.float64), {
            "R": 0.0,
            "I": 0.0,
            "h": 0.0,
            "u_m": 0.0,
        }

    # Set the other state variable
    hu_m = U[1]

    # Yield primitive variable
    u_m = hu_m / h

    # Build rainfall-infiltration mass production source term
    S = np.array([
        R - I,
        (R - I) * u_m,
    ], dtype=np.float64)

    # Set up diagnostics for debugging
    diagnostics = {
        "R": R,
        "I": I,
        "h": h,
        "u_m": u_m,
    }

    # Output
    return S, diagnostics

def compute_friction_matrix_n0(
        values,
        R,
        I,
        mixing_friction_model,
        viscosity,
        slip_length,
        context = None,
        eps : float = 1e-14,
    ):
    """
    Compute the raw friction/mixing block P(U) for the N=0 SWE model with 
    rainfall and infiltration.

    State:
        U = [h, h * u_m]

    Effective mixing-friction coefficients are now evaluated locally as:
        f_R = alpha_R * R
        f_I = alpha_I * max(0, -I)

    Source:
        P(U) = [
        0, 
        (f_R + f_I + nu/lambda) * u_m 
        ]

    Parameters
    ----------
    values : array_like
        Conserved state variables [h, h * u_m].
    R : float
        Rainfall rate.
    I : float
        Infiltration or Exfiltration rate.
    mixing_friction_model : object,
        Chosen closure for the mixing-induced friction coefficients f_R, f_I.
    viscosity : float
        Dynamic viscosity for Navier slip law.
    slip_length : float
        Slip length for Navier slip law.
    context : SourceContext
        Runtime source context.
    eps : float
        Small tolerance to avoid division by zero.

    Notes
    -----
    The solver should use S_total = S_{R, I}(U) - P(U)
    """
    # Validate state size
    U = np.asarray(values, dtype=np.float64)
    if U.shape[0] != 2:
        raise ValueError("N=0 SWME system expects state vector of length 2.")
    
    # Extract height
    h = U[0]

    # Dry-cell treatment
    if h <= eps:
        return np.zeros(3, dtype=np.float64), {
            "h" : 0.0,
            "u_m" : 0.0,
        }
    
    # Validate slip length
    if slip_length <= eps:
        raise ValueError(
            "The current friction implementation uses nu / lambda explicitly,  " \
            "so it requires slip_length > 0."
        )
    
    # Set the other state variable
    hu_m = U[1]

    # Yield the primitive variable
    u_m = hu_m / h

    # Compute the mixing friction coefficients from the closure
    f_R, f_I = mixing_friction_model.evaluate(
        R = R,
        I = I,
        context = context,
        values = U
    )

    # Define the composite friction coefficients
    # There are of the form Navier Slip Law + Mixing Induced Friction

    # One is the total sum, the other looks like a skewness condition
    coeff = f_R + f_I + viscosity / slip_length

    # Build the composite friction vector as a numpy array
    P = np.array([
        0.0,
        coeff * u_m,
    ], dtype=np.float64)

    # Diagnostics for debugging
    diagnostics = {
        "h": h,
        "u_m": u_m,
        "R" : float(R),
        "I" : float(I),
        "f_R" : float(f_R),
        "f_I" : float(f_I),
        "mixing_friction_model" : mixing_friction_model.__class__.__name__,
    }

    return P, diagnostics

# ============================================================================ #
# N = 1 
# ============================================================================ #

def compute_recharge_source_n1(
        values,
        rainfall,
        infiltration_model,
        context,
        eps : float = 1e-14,
    ):
    """
    Compute the pure rainfall-infiltration source term S_{R, I}(U) for the 
    first order SWME system.

    State:
        U = [h, h * u_m, h * a_1]

    Source:
        S_{R, I}(U) = [
            R - I, 
            (R - I) * u_m - (R + I) * a_1,
            2 * (R - I) * a_1
        ]

    Parameters
    ----------
    values : array_like
        Conserved state variables [h, h * u_m, h * a_1].
    rainfall : float or callable
        Rainfall rate R. If callable, it is evaluated as rainfall(context).
    infiltration_model : object
        A customly encoded infiltration model. Must provide a method
            rate(t, rainfall, h, dt) or similar
    context : SourceContext
        Runtime source context.
    eps : float
        Small tolerance to avoid division by zero.

    Returns
    -------
    np.ndarray
        Recharge source vector of shape (3,).
    dict
        Diagnostics: R, I, h, u_m, a_1
    """
    # First evaluate rainfall and infiltration, with dry-cell treatment
    U, h, R, I, dry = _evaluate_rainfall_and_infiltration(
        values = values,
        rainfall = rainfall,
        infiltration_model = infiltration_model,
        context = context,
        expected_size = 3,
        eps = eps,
    )

    # If dry, return zero source and diagnostics with zeros
    if dry:
        return np.zeros(3, dtype = np.float64), {
            "R": 0.0,
            "I": 0.0,
            "h": 0.0,
            "u_m": 0.0,
            "a_1": 0.0,
        }

    # Set the other two state variables
    hu_m = U[1]
    ha_1 = U[2]

    # Yield primitive variables
    u_m = hu_m / h
    a1 = ha_1 / h

    # Build rainfall-infiltration mass production source term
    S = np.array([
        R - I,
        (R - I) * u_m - (R + I) * a1,
        2 * (R - I) * a1,
    ], dtype=np.float64)

    # Set up diagnostics for debugging
    diagnostics = {
        "R": R,
        "I": I,
        "h": h,
        "u_m": u_m,
        "a_1": a1,
    }

    # Output
    return S, diagnostics

def compute_friction_matrix_n1(
        values,
        R,
        I,
        mixing_friction_model,
        viscosity,
        slip_length,
        context = None,
        eps : float = 1e-14,
    ):
    """
    Compute the raw friction/mixing block P(U) for the N=1 SWME model with 
    rainfall and infiltration.

    State:
        U = [h, h * u_m, h * a_1]

    Effective mixing-friction coefficients are now evaluated locally as:
        f_R = alpha_R * R
        f_I = alpha_I * max(0, -I)

    Source:
        P(U) = [
        0, 
        (f_R + f_I + nu/lambda) * u_m - (f_R - f_I - nu/lambda) * a_1,
        -3 (f_R - f_I - nu/lambda) * u_m + ( 3(f_R + f_I + nu/lambda) + 12 nu / h) * a_1
        ]

    Parameters
    ----------
    values : array_like
        Conserved state variables [h, h * u_m, h * a_1, h * a_2].
    R : float
        Rainfall rate.
    I : float
        Infiltration or Exfiltration rate.
    mixing_friction_model : object,
        Chosen closure for the mixing-induced friction coefficients f_R, f_I.
    viscosity : float
        Dynamic viscosity for Navier slip law.
    slip_length : float
        Slip length for Navier slip law.
    context : SourceContext
        Runtime source context.
    eps : float
        Small tolerance to avoid division by zero.

    Notes
    -----
    The solver should use S_total = S_{R, I}(U) - P(U)
    """
    # Validate state size
    U = np.asarray(values, dtype=np.float64)
    if U.shape[0] != 3:
        raise ValueError("N=1 SWME system expects state vector of length 3.")
    
    # Extract height
    h = U[0]

    # Dry-cell treatment
    if h <= eps:
        return np.zeros(3, dtype=np.float64), {
            "h" : 0.0,
            "u_m" : 0.0,
            "a_1" : 0.0,
        }
    
    # Validate slip length
    if slip_length <= eps:
        raise ValueError(
            "The current friction implementation uses nu / lambda explicitly,  " \
            "so it requires slip_length > 0."
        )
    
    # Set the other two state variables
    hu_m = U[1]
    ha_1 = U[2]

    # Yield primitive variables
    u_m = hu_m / h
    a1 = ha_1 / h

    # Compute the mixing friction coefficients from the closure
    f_R, f_I = mixing_friction_model.evaluate(
        R = R,
        I = I,
        context = context,
        values = U
    )

    # Define the composite friction coefficients
    # There are of the form Navier Slip Law + Mixing Induced Friction

    # One is the total sum, the other looks like a skewness condition
    coeff = f_R + f_I + viscosity / slip_length
    skew = f_R - f_I - viscosity / slip_length

    # Build the composite friction vector as a numpy array
    P = np.array([
        0.0,
        coeff * u_m - skew * a1,
        -3 * skew * u_m + (3 * coeff + 12 * viscosity / h) * a1,
    ], dtype=np.float64)

    # Diagnostics for debugging
    diagnostics = {
        "h": h,
        "u_m": u_m,
        "a_1": a1,
        "R" : float(R),
        "I" : float(I),
        "f_R" : float(f_R),
        "f_I" : float(f_I),
        "mixing_friction_model" : mixing_friction_model.__class__.__name__,
    }

    return P, diagnostics

# ============================================================================ #
# N = 2
# ============================================================================ #

def compute_recharge_source_n2(
        values,
        rainfall,
        infiltration_model,
        context,
        eps : float = 1e-14,
    ):
    """
    Pure rainfall-infiltration source term S_{R, I}(U) for N = 2.

    State:
        U = [h, h * u_m, h * a_1, h * a_2]

    Source:
        S_{R, I}(U) = [
            R - I, 
            (R - I) * u_m - (R + I) * a_1 + (R - I) * a_2,
            2 * (R - I) * a_1 - 3 * (R + I) * a_2,
            3 * (R - I) * a_2
        ]
    
    Parameters
    ----------
    values : array_like
        Conserved state variables [h, h * u_m, h * a_1, h * a_2].
    rainfall : float or callable
        Rainfall rate R. If callable, it is evaluated as rainfall(context).
    infiltration_model : object
        A customly encoded infiltration model. Must provide a method
        rate(t, rainfall, h, dt) or similar
    context : SourceContext
        Runtime source context.
    eps : float
        Small tolerance to avoid division by zero.
    
    Returns
    -------
    np.ndarray
        Recharge source vector of shape (4,).
    dict
        Diagnostics: R, I, h, u_m, a_1, a_2
    """
    # First evaluate rainfall and infiltration, with dry-cell treatment
    U, h, R, I, dry = _evaluate_rainfall_and_infiltration(
        values = values,
        rainfall = rainfall,
        infiltration_model = infiltration_model,
        context = context,
        expected_size = 4,
        eps = eps,
    )

    # If dry, return zero source and diagnostics with zeros
    if dry:
        return np.zeros(4, dtype = np.float64), {
            "R": 0.0,
            "I": 0.0,
            "h": 0.0,
            "u_m": 0.0,
            "a_1": 0.0,
            "a_2": 0.0,
        }
    
    # Set the other three state variables
    hu_m = U[1]
    ha_1 = U[2]
    ha_2 = U[3]

    # Yield primitive variables
    u_m = hu_m / h
    a1 = ha_1 / h
    a2 = ha_2 / h

    # Build rainfall-infiltration mass production source term
    S = np.array([
        R - I,
        (R - I) * u_m - (R + I) * a1 + (R - I) * a2,
        2.0 * (R - I) * a1 - 3.0 * (R + I) * a2,
        3.0 * (R - I) * a2,
    ], dtype=np.float64)

    # Set up diagnostics for debugging
    diagnostics = {
        "R": R,
        "I": I,
        "h": h,
        "u_m": u_m,
        "a_1": a1,
        "a_2": a2,
    }

    # Output
    return S, diagnostics

def compute_friction_matrix_n2(
        values,
        R,
        I,
        mixing_friction_model,
        viscosity,
        slip_length,
        context = None,
        eps : float = 1e-14,
    ):
    """
    Compute the raw friction/mixing block P(U) for the N=2 SWME model
    with rainfall and infiltration.

    State:
        U = [h, h * u_m, h * a_1, h * a_2]

    Effective mixing-friction coefficients are now evaluated locally as:
        f_R = alpha_R * R
        f_I = alpha_I * max(0, -I)

    Source:
        P(U) = [
        0,
        coeff * u_m - skew * a_1 + coeff * a_2,
        -3 * skew * u_m + (3 * coeff + 12 * viscosity / h) * a_1 - 3 * skew * a_2,
        5 * coeff * u_m - 5 * skew * a_1 + (5 * coeff + 60 * viscosity / h) * a_2
        ]

    where
        coeff = f_R + f_I + viscosity / slip_length
        skew = f_R - f_I - viscosity / slip_length

    Parameters
    ----------
    values : array_like
        Conserved state variables [h, h * u_m, h * a_1, h * a_2].
    R : float
        Rainfall rate.
    I : float
        Infiltration or Exfiltration rate.
    mixing_friction_model : object,
        Chosen closure for the mixing-induced friction coefficients f_R, f_I.
    viscosity : float
        Dynamic viscosity for Navier slip law.
    slip_length : float
        Slip length for Navier slip law.
    context : SourceContext
        Runtime source context.
    eps : float
        Small tolerance to avoid division by zero.

    Returns
    -------
    np.ndarray
        Friction source vector of shape (4,).
    dict
        Diagnostics: h, u_m, a_1, a_2, f_R, f_I
    """
    # Validate state size
    U = np.asarray(values, dtype=np.float64)
    if U.shape[0] != 4:
        raise ValueError("N=2 SWME system expects state vector of length 4.")
    
    # Extract height
    h = U[0]

    # Dry-cell treatment
    if h <= eps:
        return np.zeros(4, dtype=np.float64), {
            "h" : 0.0,
            "u_m" : 0.0,
            "a_1" : 0.0,
            "a_2" : 0.0,
        }
    
    if slip_length <= eps:
        raise ValueError(
            "The current friction implementation uses nu / lambda explicitly,  "\
            "so it requires slip_length > 0."
        )
    
    # Set the other three state variables
    hu_m = U[1]
    ha_1 = U[2]
    ha_2 = U[3]

    # Yield primitive variables
    u_m = hu_m / h
    a1 = ha_1 / h
    a2 = ha_2 / h

    # Compute the mixing friction coefficients from the closure
    f_R, f_I = mixing_friction_model.evaluate(
        R = R,
        I = I,
        context = context,
        values = U
    )

    # Define the composite friction coefficients
    coeff = f_R + f_I + viscosity / slip_length
    skew = f_R - f_I - viscosity / slip_length

    # Build the composite friction vector as a numpy array
    P = np.array([
        0.0,
        coeff * u_m - skew * a1 + coeff * a2,
        -3.0 * skew * u_m + (3.0 * coeff + 12.0 * viscosity / h) * a1 - 3.0 * skew * a2,
        5.0 * coeff * u_m - 5.0 * skew * a1 + (5.0 * coeff + 60.0 * viscosity / h) * a2,
    ], dtype=np.float64)

    # Diagnostics for debugging
    diagnostics = {
        "h" : h,
        "u_m" : u_m,
        "a_1" : a1,
        "a_2" : a2,
        "R" : float(R),
        "I" : float(I),
        "f_R" : float(f_R),
        "f_I" : float(f_I),
        "mixing_friction_model" : mixing_friction_model.__class__.__name__,
    }

    return P, diagnostics

# ============================================================================ #
# Dispatchers
# ============================================================================ #

def compute_recharge_source(
        order,
        values,
        rainfall,
        infiltration_model,
        context,
        eps : float = 1e-14,
    ):
    """
    Dispatch pure recharge source S_{R, I}(U) by moment order.
    """
    if order == 0:
        return compute_recharge_source_n0(
            values = values,
            rainfall = rainfall,
            infiltration_model = infiltration_model,
            context = context,
            eps = eps,
        )
    elif order == 1:
        return compute_recharge_source_n1(
            values = values,
            rainfall = rainfall,
            infiltration_model = infiltration_model,
            context = context,
            eps = eps,
        )
    elif order == 2:
        return compute_recharge_source_n2(
            values = values,
            rainfall = rainfall,
            infiltration_model = infiltration_model,
            context = context,
            eps = eps,
        )
    else:
        raise NotImplementedError(
            f"Recharge source currently implemented only for N=1 and N=2."
        )

def compute_friction_matrix(
        order,
        values,
        R,
        I,
        mixing_friction_model,
        viscosity,
        slip_length,
        context = None,
        eps : float = 1e-14,
    ):
    """
    Dispatch friction/mixing block P(U) by moment order.
    """
    if order == 0:
        return compute_friction_matrix_n0(
            values = values,
            R = R,
            I = I,
            mixing_friction_model = mixing_friction_model,
            viscosity = viscosity,
            slip_length = slip_length,
            context = context,
            eps = eps,
        )
    elif order == 1:
        return compute_friction_matrix_n1(
            values = values,
            R = R,
            I = I,
            mixing_friction_model = mixing_friction_model, 
            viscosity = viscosity,
            slip_length = slip_length,
            context = context,
            eps = eps,
        )
    elif order == 2:
        return compute_friction_matrix_n2(
            values = values,
            R = R,
            I = I,
            mixing_friction_model = mixing_friction_model, 
            viscosity = viscosity,
            slip_length = slip_length,
            context = context,
            eps = eps,
        )
    else:
        raise NotImplementedError(
            f"Friction source currently implemented only for N=1 and N=2."
        )

def compute_total_source(
        order,
        values,
        rainfall,
        infiltration_model,
        context,
        mixing_friction_model,
        viscosity,
        slip_length,
        eps : float = 1e-14,
        ):
    """
    Compute the full recharge source:
        S_total(U) = S_{R, I}(U) - P(U)
    for N = 1 or N = 2.

    The recharge block S_{R, I}(U) is built first, yielding the local rainfall
    R and infiltration I. These local rates are then used to evaluate the 
    mixing-friction law and construct P(U).
    """
    # First build the recharge mass production vector
    S, recharge_diagnostics = compute_recharge_source(
        order = order,
        values = values,
        rainfall = rainfall,
        infiltration_model = infiltration_model,
        context = context,
        eps = eps,
    )

    # Then build the composite friction vector
    P, friction_diagnostics = compute_friction_matrix(
        order = order,
        values = values,
        R = recharge_diagnostics["R"],
        I = recharge_diagnostics["I"],
        mixing_friction_model = mixing_friction_model,
        viscosity = viscosity,
        slip_length = slip_length,
        eps = eps,
    )

    # Combine diagnostics into one dictionary
    diagnostics = {
        "recharge": recharge_diagnostics,
        "friction": friction_diagnostics,
    }

    return S - P, diagnostics