"""
model.py

Simulation engine for reciprocal-space maps (RSMs).

Given a meshgrid (xs, ys) and a SimulationParameters object,
returns the simulated RSM for further use in fitting and plotting.
"""

import numpy as np

from geometry import (
    plane_basis,
    xy_in_plane,
    projected_peaks,
)

# Conversion factor between FWHM and sigma
FWHM_TO_SIGMA = 2 * np.sqrt(2 * np.log(2))


def rotated_gaussian(
    xs,
    ys,
    x0,
    y0,
    theta,
    fwhm_long,
    fwhm_short,
    amplitude,
):
    """
    Generates a rotated elliptical Gaussian.

    Parameters
    ----------
    xs, ys : ndarray
        Meshgrid coordinates.

    x0, y0 : float
        Centre of the Gaussian.

    theta : float
        Rotation angle (radians).

    fwhm_long : float
        FWHM along the long axis.

    fwhm_short : float
        FWHM along the short axis.

    amplitude : float
        Peak intensity.

    Returns
    -------
    ndarray
        Gaussian evaluated on the grid.
    """

    sigma_long = fwhm_long / FWHM_TO_SIGMA
    sigma_short = fwhm_short / FWHM_TO_SIGMA

    xr = (xs - x0) * np.cos(theta) + (ys - y0) * np.sin(theta)

    yr = -(xs - x0) * np.sin(theta) + (ys - y0) * np.cos(theta)

    return amplitude * np.exp(
        -0.5 * (xr / sigma_long) ** 2
        -0.5 * (yr / sigma_short) ** 2
    )


def simulate_image(xs, ys, params):
    """
    Simulates a reciprocal-space intensity map.

    Parameters
    ----------
    xs, ys : ndarray
        Meshgrid coordinates.

    params : SimulationParameters

    Returns
    -------
    image : ndarray
        Simulated intensity map.
    """

    # -----------------------------------------------------
    # Project satellite peaks
    # -----------------------------------------------------

    if params.Sub_peak is None:

        peak1, peak2 = projected_peaks(params)

        substrate = None

    else:

        peak1, peak2, substrate = projected_peaks(params)

    # -----------------------------------------------------
    # Build plane basis
    # -----------------------------------------------------

    _, u, v = plane_basis(
        params.Qin,
        params.Qout,
    )

    Qin_length = np.linalg.norm(params.Qin)
    Qout_length = np.linalg.norm(params.Qout)

    # -----------------------------------------------------
    # Convert peaks to 2D plane coordinates
    # -----------------------------------------------------

    X1, Y1 = xy_in_plane(peak1, u, v)
    X2, Y2 = xy_in_plane(peak2, u, v)

    X1 /= Qin_length
    X2 /= Qin_length

    Y1 /= Qout_length
    Y2 /= Qout_length

    theta1 = np.arctan2(Y1, X1)
    theta2 = np.arctan2(Y2, X2)

    theta1 += params.shift_angle
    theta2 += params.shift_angle

    # -----------------------------------------------------
    # Background
    # -----------------------------------------------------

    image = np.full_like(xs, params.background, dtype=float)

    # -----------------------------------------------------
    # Satellite Peak 1
    # -----------------------------------------------------
    
    image += rotated_gaussian(
        xs,
        ys,
        X1,
        Y1,
        theta1,
        params.film_long,
        params.film_short,
        params.film_intensity,
    )

    # -----------------------------------------------------
    # Satellite Peak 2
    # -----------------------------------------------------

    image += rotated_gaussian(
        xs,
        ys,
        X2,
        Y2,
        theta2,
        params.film_long,
        params.film_short,
        params.film_intensity,
    )

    # -----------------------------------------------------
    # Substrate Peak
    # -----------------------------------------------------

    if substrate is not None:

        XS, YS = xy_in_plane(substrate, u, v)

        XS /= Qin_length
        YS /= Qout_length

        thetaS = np.arctan2(YS, XS)

        thetaS += params.shift_angle_sub

        image += rotated_gaussian(
            xs,
            ys,
            XS,
            YS,
            thetaS,
            params.substrate_long,
            params.substrate_short,
            params.sub_intensity,
        )

    return image


def generate_grid(params, npoints=100):
    """
    Generates a meshgrid defined by the input In_Range and Out_Range parameters.

    Returns
    -------
    xs, ys
    """

    xs, ys = np.meshgrid(
        np.linspace(
            params.In_Range[0],
            params.In_Range[1],
            npoints,
        ),
        np.linspace(
            params.Out_Range[0],
            params.Out_Range[1],
            npoints,
        ),
        indexing="ij",
    )

    return xs, ys