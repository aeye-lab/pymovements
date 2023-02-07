"""
This module holds the base classes Screen and Experiment.
"""
from __future__ import annotations

import numpy as np

from pymovements.transforms import pix2deg
from pymovements.transforms import pos2vel
from pymovements.utils import checks
from pymovements.utils.decorators import auto_str


@auto_str
class Screen:
    """
    Screen class for holding screen properties and transforming pixel
    coordinates to degrees of visual angle.

    Attributes
    ----------
    width_px : int
        Screen width in pixels
    height_px : int
        Screen height in pixels
    width_cm : float
        Screen width in centimeters
    height_cm : float
        Screen height in centimeters
    distance_cm : float
        Eye-to-screen distance in centimeters
    origin : str
        Specifies the screen location of the origin of the pixel coordinate system.
    x_max_dva : float
        Maximum screen x-coordinate in degrees of visual angle
    y_max_dva : float
        Minimum screen y-coordinate in degrees of visual angle
    x_min_dva : float
        Maximum screen x-coordinate in degrees of visual angle
    y_min_dva : float
        Minimum screen y-coordinate in degrees of visual angle

    """
    def __init__(
        self,
        width_px: int,
        height_px: int,
        width_cm: float,
        height_cm: float,
        distance_cm: float,
        origin: str,
    ):
        """
        Initializes Screen.

        Parameters
        ----------
        width_px : int
            Screen width in pixels
        height_px : int
            Screen height in pixels
        width_cm : float
            Screen width in centimeters
        height_cm : float
            Screen height in centimeters
        distance_cm : float
            Eye-to-screen distance in centimeters
        origin : str
            Specifies the screen location of the origin of the pixel coordinate system.

        Examples
        --------
        >>> screen = Screen(
        ...     width_px=1280,
        ...     height_px=1024,
        ...     width_cm=38,
        ...     height_cm=30,
        ...     distance_cm=68,
        ...     origin='lower left',
        ... )
        >>> print(screen)  # doctest: +NORMALIZE_WHITESPACE
        Screen(width_px=1280, height_px=1024, width_cm=38, height_cm=30, distance_cm=68,
        origin=lower left, x_max_dva=15.60, y_max_dva=12.43, x_min_dva=-15.60, y_min_dva=-12.43)

        """
        checks.check_no_zeros(width_px, "width_px")
        checks.check_no_zeros(height_px, "height_px")
        checks.check_no_zeros(width_cm, "width_cm")
        checks.check_no_zeros(height_cm, "height_cm")
        checks.check_no_zeros(distance_cm, "distance_cm")

        self.width_px = width_px
        self.height_px = height_px
        self.width_cm = width_cm
        self.height_cm = height_cm
        self.distance_cm = distance_cm
        self.origin = origin

        # calculate screen boundary coordinates in degrees of visual angle
        self.x_max_dva = pix2deg(width_px-1, width_px, width_cm, distance_cm, origin=origin)
        self.y_max_dva = pix2deg(height_px-1, height_px, height_cm, distance_cm, origin=origin)
        self.x_min_dva = pix2deg(0, width_px, width_cm, distance_cm, origin=origin)
        self.y_min_dva = pix2deg(0, height_px, height_cm, distance_cm, origin=origin)

    def pix2deg(
            self,
            arr: float | list[float] | list[list[float]] | np.ndarray,
    ) -> np.ndarray:
        """
        Converts pixel screen coordinates to degrees of visual angle.

        Parameters
        ----------
        arr : float, array_like
            Pixel coordinates to transform into degrees of visual angle

        Returns
        -------
        degrees_of_visual_angle : np.ndarray
            Coordinates in degrees of visual angle

        Raises
        ------
        ValueError
            If positions aren't two-dimensional.

        Examples
        --------
        >>> screen = Screen(
        ...     width_px=1280,
        ...     height_px=1024,
        ...     width_cm=38,
        ...     height_cm=30,
        ...     distance_cm=68,
        ...     origin='lower left',
        ... )
        """
        return pix2deg(
            arr=arr,
            screen_px=(self.width_px, self.height_px),
            screen_cm=(self.width_cm, self.height_cm),
            distance_cm=self.distance_cm,
            origin=self.origin,
        )


class Experiment:
    """
    Experiment class for holding experiment properties.

    Attributes
    ----------

    screen : Screen
        Screen object for experiment
    sampling_rate : float
        Sampling rate in Hz
    """

    def __init__(
        self, screen_width_px: int, screen_height_px: int,
        screen_width_cm: float, screen_height_cm: float,
        distance_cm: float, origin: str, sampling_rate: float,
    ):
        """
        Initializes Experiment.

        Parameters
        ----------

        screen_width_px : int
            Screen width in pixels
        screen_height_px : int
            Screen height in pixels
        screen_width_cm : float
            Screen width in centimeters
        screen_height_cm : float
            Screen height in centimeters
        distance_cm : float
            Eye-to-screen distance in centimeters
        origin : str
            Specifies the screen location of the origin of the pixel coordinate system.
        sampling_rate : float
            Sampling rate in Hz

        """
        self.screen = Screen(
            width_px=screen_width_px,
            height_px=screen_height_px,
            width_cm=screen_width_cm,
            height_cm=screen_height_cm,
            distance_cm=distance_cm,
            origin=origin,
        )
        self.sampling_rate = sampling_rate

    def pos2vel(
        self,
        arr: list[float] | list[list[float]] | np.ndarray,
        method: str = 'smooth',
        **kwargs,
    ) -> np.ndarray:
        """Compute velocity time series from 2-dimensional position time series.

        Methods 'smooth', 'neighbors' and 'preceding' are adapted from
            Engbert et al.: Microsaccade Toolbox 0.9.

        Parameters
        ----------
        arr : array_like
            Continuous 2D position time series
        method : str
            Computation method. See :func:`~transforms.pos2vel` for details, default: smooth.
        kwargs: dict
            Additional keyword arguments used for savitzky golay method.

        Returns
        -------
        velocities : array_like
            Velocity time series in input_unit / sec

        Raises
        ------
        ValueError
            If selected method is invalid, input array is too short for the
            selected method or the sampling rate is below zero

        """
        return pos2vel(arr=arr, sampling_rate=self.sampling_rate, method=method, **kwargs)
