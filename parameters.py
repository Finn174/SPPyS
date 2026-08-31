"""
parameters.py

Defines the simulation parameter dataclass.
"""

from dataclasses import dataclass, fields
from typing import Optional
import numpy as np


@dataclass
class SimulationParameters:
    """Container for all simulation parameters."""

    # Reciprocal-space geometry
    Qin: np.ndarray
    Qout: np.ndarray

    Bragg_peak: np.ndarray
    prop_vector: np.ndarray

    # Splitting
    delta: float

    # Intensities
    background: float
    film_intensity: float
    sub_intensity: float = 0.0

    # Film resolution ellipse
    film_long: float = 0.03
    film_short: float = 0.01

    # Substrate resolution ellipse
    substrate_long: float = 0.03
    substrate_short: float = 0.01

    # Peak rotation angle (in radians)
    shift_angle: float = 0.0
    shift_angle_sub: float = 0.0

    # Peak shifts in the detector plane (in reciprocal space units)
    shift_x: float = 0.0
    shift_y: float = 0.0
    shift_x_sub: float = 0.0
    shift_y_sub: float = 0.0

    # Instrument
    sgl_width: float = 2.0

    # Plot ranges
    In_Range: Optional[list] = None
    Out_Range: Optional[list] = None

    # Optional substrate peak
    Sub_peak: Optional[np.ndarray] = None

    # Optional color scale limits for plotting
    axis_compress: Optional[float] = 1

    # Optional axis labels for plotting
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None

    # Optional bounds for fitting
    delta_bounds: Optional[list] = None
    background_bounds: Optional[list] = None
    film_intensity_bounds: Optional[list] = None
    sub_intensity_bounds: Optional[list] = None
    film_long_bounds: Optional[list] = None
    film_short_bounds: Optional[list] = None
    substrate_long_bounds: Optional[list] = None
    substrate_short_bounds: Optional[list] = None
    shift_angle_bounds: Optional[list] = None
    shift_angle_sub_bounds: Optional[list] = None
    shift_x_bounds: Optional[list] = None
    shift_y_bounds: Optional[list] = None
    shift_x_sub_bounds: Optional[list] = None
    shift_y_sub_bounds: Optional[list] = None


    def update(self, **kwargs):
        """
        Update one or more parameters by name.

        Example
        -------
        params.update(
            delta=0.045,
            film_intensity=500
        )
        """

        valid_fields = {f.name for f in fields(self)}

        for key, value in kwargs.items():

            if key not in valid_fields:
                raise AttributeError(f"Unknown parameter '{key}'")

            setattr(self, key, value)

    def as_vector(self, parameter_names):
        """
        Convert selected parameters into a NumPy vector.

        Example
        -------
        params.as_vector([
            "delta",
            "film_intensity",
            "background"
        ])
        """

        return np.array(
            [getattr(self, name) for name in parameter_names],
            dtype=float,
        )
    

    def from_vector(self, vector, parameter_names):
        """
        Update parameters from a fitted vector.
        
        Only scalar parameters can be updated.
        
        Example
        -------
        params.from_vector(
            result.x,
            [
                "delta",
                "film_intensity",
                "background"
            ]
        )
        """

        for name, value in zip(parameter_names, vector):
            #Check that the parameter exists
            if not hasattr(self, name):
                raise AttributeError(f"Unknown parameter '{name}'")
        
            current = getattr(self, name)
        
            #Prevent accidentally fitting arrays
            if np.ndim(current) != 0:
                raise TypeError(
                    f"'{name}' is not a scalar parameter "
                    "and cannot be updated from a fit vector."
                )
        
            setattr(self, name, float(value))