"""
fit.py

Least-squares fitting routines for SPPyS.

Uses scipy.optimize.least_squares.
"""

import numpy as np

from scipy.optimize import least_squares

from geometry import splitting_is_resolvable

from model import simulate_image


# ========================================================
# Bounds conversion
# ========================================================

def convert_bounds(bounds, fit_parameters):
    """
    Convert dictionary bounds into
    scipy.optimize.least_squares format.
    """

    if bounds is None:
        return (-np.inf, np.inf)

    lower = []
    upper = []

    for name in fit_parameters:

        if name not in bounds:
            raise ValueError(
                f"No bound supplied for '{name}'"
            )

        low, high = bounds[name]

        if low is None:
            low = -np.inf

        if high is None:
            high = np.inf

        lower.append(low)
        upper.append(high)

    return (
        np.array(lower),
        np.array(upper)
    )


# ========================================================
# Residual calculation
# ========================================================

def residuals(
    current_values,
    params,
    fit_parameters,
    xs,
    ys,
    experimental,
    mask=None,
):
    params.from_vector(current_values, fit_parameters)
    
    simulated = simulate_image(xs, ys, params)
    
    if mask is not None:
        difference = simulated[mask] - experimental[mask]
        exp_data = experimental[mask]
    else:
        difference = simulated - experimental
        exp_data = experimental
    
    if not np.all(np.isfinite(difference)):
        raise ValueError(
            "Residual contains NaN or inf values."
        )
    
    # Weighting: sqrt(1/variance) = 1/sqrt(N)
    weights = 1.0 / np.sqrt(exp_data)
    
    weighted_difference = difference * weights
    
    return weighted_difference.ravel()


def jacobian_function(current_values, params, fit_parameters, xs, ys, experimental, mask):
    """
    Calculates the Jacobian using a central difference method.
    """

    jac = np.zeros((np.sum(mask), len(fit_parameters)))
    
    for i in range(len(fit_parameters)):
        epsilon = max(1e-8, 1e-5 * abs(current_values[i]))
        
        vals_plus = current_values.copy()
        vals_plus[i] += epsilon
        residuals_plus = residuals(vals_plus, params, fit_parameters, xs, ys, experimental, mask)
        
        vals_minus = current_values.copy()
        vals_minus[i] -= epsilon
        residuals_minus = residuals(vals_minus, params, fit_parameters, xs, ys, experimental, mask)
        
        jac[:, i] = (residuals_plus - residuals_minus) / (2 * epsilon)
    
    return jac


# ========================================================
# Run fit
# ========================================================

def fit_image(
    params,
    xs,
    ys,
    experimental,
    fit_parameters,
    initial_guess=None,
    bounds=None,
):
    """
    Fits simulation to experimental data.

    NaN values in experimental data are ignored.
    """


    # --------------------------------
    # Build valid data mask
    # --------------------------------

    mask = np.isfinite(
        experimental
    )


    if np.sum(mask) == 0:

        raise ValueError(
            "Experimental image contains no valid pixels."
        )


    # --------------------------------
    # Initial parameter vector
    # --------------------------------

    if initial_guess is None:

        initial_guess = params.as_vector(
            fit_parameters
        )


    # --------------------------------
    # Bounds
    # --------------------------------

    bounds = convert_bounds(
        bounds,
        fit_parameters,
    )

    # --------------------------------
    # Optimisation
    # --------------------------------

    result = least_squares(

        residuals,

        initial_guess,

        jac=jacobian_function,

        args=(

            params,

            fit_parameters,

            xs,

            ys,

            experimental,

            mask,

        ),

        bounds=bounds,

        verbose=1,

        x_scale='jac',

    )

    # Update params with final solution

    params.from_vector(
        result.x,
        fit_parameters,
    )

    return result


# ========================================================
# Statistics
# ========================================================

def calculate_statistics(
    result,
    experimental,
    fitted=None,
):
    """
    Calculates relevant fit statistics.

    Parameters
    ----------
    result :
        scipy OptimizeResult

    experimental :
        Experimental intensity map

    fitted :
        Final simulated intensity map
    """

    residual = result.fun

    chi_squared = np.sum(
        residual**2
    )

    n_points = residual.size

    dof = (
        n_points
        -
        len(result.x)
    )

    reduced_chi_squared = (
        chi_squared / dof
        if dof > 0
        else np.nan
    )

    stats = {
        "chi_squared":
            chi_squared,

        "reduced_chi_squared":
            reduced_chi_squared,

        "n_points":
            n_points,
    }

    return stats
