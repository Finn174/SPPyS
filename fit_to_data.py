"""
fit_to_data.py

Functions for fitting a simulated reciprocal-space map
to experimental data.
"""

import sys
import os
import numpy as np

from geometry import splitting_is_resolvable

from data_io import (
    load_parameters,
    load_experimental_data,
    save_parameters,
)

from model import simulate_image

from fit import (
    fit_image,
    calculate_statistics,
)

from plotting import (
    plot_fit_results,
)


def run_fit(
    params,
    data_file,
    fit_parameters,
    bounds=None,
):
    """
    Performs one fitting instance.

    Returns
    -------
    result
        scipy optimisation result

    params
        Updated parameter dictionary with fitted values
    """

    # --------------------------------------------------
    # Load experimental data
    # --------------------------------------------------

    xs, ys, experimental = load_experimental_data(
        data_file,
        Qin=params.Qin,
        Qout=params.Qout,
    )

    # --------------------------------------------------
    # Generate initial simulation
    # --------------------------------------------------
    
    # Experimental coordinates define the fitting grid.
    # The simulation is evaluated directly on these points.

    initial_image = simulate_image(
        xs,
        ys,
        params,
    )
    
    # --------------------------------------------------
    # Perform fit
    # --------------------------------------------------

    result = fit_image(
        params,
        xs,
        ys,
        experimental,
        fit_parameters,
        bounds=bounds,
    )

    # --------------------------------------------------
    # Generate fitted simulation
    # --------------------------------------------------

    fitted_image = simulate_image(
        xs,
        ys,
        params,
    )

    params.update(In_Range=(np.min(xs), np.max(xs)))
    params.update(Out_Range=(np.min(ys), np.max(ys)))

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    stats = calculate_statistics(
        result,
        experimental,
        fitted_image,
    )

    print("\n" + "Reduced χ² =", stats["reduced_chi_squared"])

    red_chi2 = stats["reduced_chi_squared"]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    output_filename = data_file.split('/')[-1].replace('.dat', '')

    plot_fit_results(
        params,
        xs,
        ys,
        experimental,
        fitted_image,
        output_filename,
        red_chi2,
    )

    return result, params


def get_parameter_errors(result, fit_parameters, n_data_points=None):
    """
    Extracts parameter errors from least_squares result.
    """
    
    # Try to use result.cov_x if available
    if hasattr(result, 'cov_x') and result.cov_x is not None:
        cov_matrix = result.cov_x
    else:
        # Manual computation as fallback
        residuals = result.fun
        jacobian = result.jac
        n_params = len(fit_parameters)
        
        if n_data_points is None:
            n_data_points = len(residuals)
        
        dof = n_data_points - n_params
        residual_variance = np.sum(residuals**2) / dof
        
        try:
            cov_matrix = residual_variance * np.linalg.inv(jacobian.T @ jacobian)
        except np.linalg.LinAlgError:
            print("Warning: Singular Jacobian, using pseudo-inverse")
            cov_matrix = residual_variance * np.linalg.pinv(jacobian.T @ jacobian)
    
    std_errors = np.sqrt(np.diag(cov_matrix))

    # Print results
    print("\n" + "=" * 70)
    print("PARAMETER ERRORS")
    print("=" * 70)
    
    for i, param_name in enumerate(fit_parameters):
        param_value = result.x[i]
        std_error = std_errors[i]
        relative_error = std_error / abs(param_value) * 100 if param_value != 0 else np.inf
        
        print(f"{param_name:15s} = {param_value:.8f} ± {std_error:.8f}  ({relative_error:.2f}%)")
    
    print("=" * 70)

    return cov_matrix, std_errors



def fit_to_data():
    """
    Runs the entire fitting procedure, saving the fitted parameters
    and a comparison figure of the experiment data and the fitted simulation.
    """

    # ----------------------------------------------------
    # Check command line input
    # ----------------------------------------------------

    if len(sys.argv) != 3:

        raise ValueError(
            "Usage: python fit_to_data.py parameter_file.json data_file.dat"
        )


    parameter_file = sys.argv[1]
    data_file = sys.argv[2]
    
    params = load_parameters(parameter_file)

    fit_parameters = [

        "delta",

        "film_intensity",

        "background",

        "sub_intensity",

        "film_long",

        "film_short",

        "shift_x",

        "shift_y",

        "shift_angle",

        "substrate_long",

        "substrate_short",

        "shift_x_sub",

        "shift_y_sub",

        "shift_angle_sub",

    ]

    # --------------------------------
    # Check if delta splitting is resolvable
    # --------------------------------

    if not splitting_is_resolvable(params):
        print("Delta splitting not resolvable in projection, removing from fit")
        fit_parameters = [p for p in fit_parameters if p != 'delta']

    if params.Sub_peak is None:
        print("No substrate peak provided, removing substrate parameters from fit")
        fit_parameters = [p for p in fit_parameters if not p.startswith('sub') and not p.startswith('shift_x_sub') and not p.startswith('shift_y_sub') and not p.startswith('shift_angle_sub')]

    bounds = {

        "delta": params.delta_bounds,

        "film_intensity": params.film_intensity_bounds,

        "background": params.background_bounds,

        "sub_intensity": params.sub_intensity_bounds,

        "film_long": params.film_long_bounds,

        "film_short": params.film_short_bounds,

        "shift_x": params.shift_x_bounds,

        "shift_y": params.shift_y_bounds,

        "shift_angle": params.shift_angle_bounds,

        "substrate_long": params.substrate_long_bounds,

        "substrate_short": params.substrate_short_bounds,

        "shift_x_sub": params.shift_x_sub_bounds,

        "shift_y_sub": params.shift_y_sub_bounds,

        "shift_angle_sub": params.shift_angle_sub_bounds,

    }

    result, fitted_parameters = run_fit(
        params,
        data_file,
        fit_parameters,
        bounds=bounds,
    )

    cov_matrix, std_errors = get_parameter_errors(
        result, 
        fit_parameters
    )

    filename = os.path.basename(parameter_file)
    output_file = f"fit_result_{filename}"

    save_parameters(
        output_file,
        fitted_parameters,
    )



if __name__ == "__main__":

    fit_to_data()
