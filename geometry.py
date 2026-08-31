"""
geometry.py

Functions for projecting reciprocal-space vectors into the
scattering plane.

All functions are independent of plotting and fitting.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


# --------------------------------------------------------
# Basic coordinate transforms
# --------------------------------------------------------

def plane_basis(Qin: np.ndarray, Qout: np.ndarray):
    """
    Constructs an orthonormal basis for the scattering plane defined by Qin and Qout.

    Returns
    -------
    n : ndarray
        Plane normal.

    u : ndarray
        Unit vector along Qin.

    v : ndarray
        Unit vector perpendicular to u in the scattering plane.
    """

    n = np.cross(Qin, Qout)

    n_norm = np.linalg.norm(n)

    if n_norm < 1e-12:
        raise ValueError(
            "Qin and Qout are parallel."
        )

    n /= n_norm

    u = Qin - np.dot(Qin, n) * n

    u /= np.linalg.norm(u)

    v = np.cross(n, u)

    return n, u, v


def xy_in_plane(point, u, v):
    """
    Converts a projected 3D point into in-plane coordinates.
    """

    return np.dot(point, u), np.dot(point, v)


# --------------------------------------------------------
# Checks
# --------------------------------------------------------

def check_scattering_plane(Qin, Qout):

    dot = np.dot(Qin, Qout)

    if abs(dot) > 1e-12:

        raise ValueError(
            "Qin and Qout must be orthogonal."
        )


def check_point_near_zone(Qin,
                          Qout,
                          peak_position,
                          sgl_width):
    """
    Ensures peak lies within the scattering plane.
    """

    n = np.cross(Qin, Qout)

    n_hat = n / np.linalg.norm(n)

    diff = peak_position - (Qin + Qout)

    signed_distance = np.dot(diff, n_hat)

    projected = peak_position - signed_distance * n_hat

    distance = np.linalg.norm(peak_position - projected)

    angle = np.degrees(np.arctan2(distance, np.linalg.norm(peak_position)))

    logger.info("Peak angle from plane = %.4f deg", angle)

    if angle > sgl_width / 2:

        raise ValueError("Peak lies outside scattering zone.")


# --------------------------------------------------------
# Projection
# --------------------------------------------------------


def peak_locations(Bragg_peak,
                   prop_vector,
                   delta):
    """
    Calculates the two satellite peak positions.
    """

    direction = prop_vector / np.linalg.norm(prop_vector)

    shift = delta * direction

    peak1 = Bragg_peak + shift
    peak2 = Bragg_peak - shift

    logger.info(
        "Satellite positions:\n%s\n%s",
        peak1,
        peak2
    )

    return peak1, peak2


def projected_peaks(params):
    """
    Calculates projected peak positions on the projection plane defined by Qin and Qout.

    Returns
    -------
    peak1
    peak2
    substrate peak (optional)
    """

    n, u, v = plane_basis(
        params.Qin,
        params.Qout
    )

    Bragg_peak_effective = (
        params.Bragg_peak
        + params.shift_x * u
        + params.shift_y * v
    )

    peak1, peak2 = peak_locations(
        Bragg_peak_effective,
        params.prop_vector,
        params.delta
    )

    if params.Sub_peak is not None:
        Sub_peak_effective = (
            params.Sub_peak
            + params.shift_x_sub * u
            + params.shift_y_sub * v
        )

    if params.Sub_peak is not None:
        return peak1, peak2, Sub_peak_effective
    else:
        return peak1, peak2


def splitting_is_resolvable(params, pixel_threshold=1e-6):
    """
    Checks if delta splitting is visible in the projected image.
    
    Even if delta is non-zero, it might split along a direction
    perpendicular to the imaging plane, making it indiscernable.
    
    Returns True if the projected separation is > threshold.
    """
    
    # Get the peak positions in 3D reciprocal space
    if params.Sub_peak is None:
        peak1, peak2 = projected_peaks(params)
        substrate = None
    else:
        peak1, peak2, substrate = projected_peaks(params)
    
    # Get the scattering plane basis vectors
    n, u, v = plane_basis(params.Qin, params.Qout)
    
    # Project both peaks onto the scattering plane (u, v coordinates)
    peak1_u, peak1_v = xy_in_plane(peak1, u, v)
    peak2_u, peak2_v = xy_in_plane(peak2, u, v)
    
    # Calculate projected separation in the scattering plane
    projected_separation = np.sqrt(
        (peak1_u - peak2_u)**2 + (peak1_v - peak2_v)**2
    )
    
    visible = projected_separation > pixel_threshold
    
    logger.info(
        f"Projected separation: {projected_separation:.6f} | "
        f"Visible: {visible} (threshold: {pixel_threshold})"
    )
    
    return visible