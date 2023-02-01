"""
This module holds the base classes Screen and Experiment.
"""
from __future__ import annotations

import numpy as np

from pymovements.transforms import pix2deg
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
        self, width_px: int, height_px: int, width_cm: float, height_cm: float, distance_cm: float,
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

        Examples
        --------
        >>> screen = Screen(
        ...     width_px=1280,
        ...     height_px=1024,
        ...     width_cm=38.0,
        ...     height_cm=30.0,
        ...     distance_cm=68.0,
        ... )
        >>> print(screen)  # doctest: +NORMALIZE_WHITESPACE
        Screen(width_px=1280, height_px=1024, width_cm=38.0, height_cm=30.0, distance_cm=68.0,
         x_max_dva=15.60, y_max_dva=12.43, x_min_dva=-15.60, y_min_dva=-12.43)

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

        # calculate screen boundary coordinates in degrees of visual angle
        self.x_max_dva = pix2deg(width_px-1, width_px, width_cm, distance_cm)
        self.y_max_dva = pix2deg(height_px-1, height_px, height_cm, distance_cm)
        self.x_min_dva = pix2deg(0, width_px, width_cm, distance_cm)
        self.y_min_dva = pix2deg(0, height_px, height_cm, distance_cm)

    def pix2deg(
            self,
            arr: float | list[float] | list[list[float]] | np.ndarray,
            center_origin: bool = True,
    ) -> np.ndarray:
        """
        Converts pixel screen coordinates to degrees of visual angle.

        Parameters
        ----------
        arr : float, array_like
            Pixel coordinates to transform into degrees of visual angle
        center_origin: bool
            Center origin to (0,0) if positions origin is in bottom left corner

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
        ...     width_cm=38.0,
        ...     height_cm=30.0,
        ...     distance_cm=68.0,
        ... )
        >>> screen.pix2deg((123.0, 865.0))
        array([-12.70732231,   8.65963972])
        >>> screen.pix2deg((123.0, 865.0), center_origin=False)
        array([ 3.07379946, 20.43909054])
        >>> screen.pix2deg((0.0))
        Traceback (most recent call last):
                        ...
        ValueError: arr should contain two-dimensional pixel coordinates
        """
        return pix2deg(
            arr=arr,
            screen_px=(self.width_px, self.height_px),
            screen_cm=(self.width_cm, self.height_cm),
            distance_cm=self.distance_cm,
            center_origin=center_origin,
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
        distance_cm: float, sampling_rate: float,
    ):
        """
        Initializes Experiment.

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
        sampling_rate : float
            Sampling rate in Hz

        Examples
        --------
        >>> experiment = Experiment(
        ...     screen_width_px=1280,
        ...     screen_height_px=1024,
        ...     screen_width_cm=38,
        ...     screen_height_cm=30,
        ...     distance_cm=68,
        ...     sampling_rate=1000.0
        ... )
        >>> print(experiment)  # doctest: +NORMALIZE_WHITESPACE
        Experiment(screen=Screen(width_px=1280, height_px=1024, width_cm=38,
         height_cm=30, distance_cm=68,
         x_max_dva=15.60, y_max_dva=12.43,
         x_min_dva=-15.60, y_min_dva=-12.43), sampling_rate=1000.0)

        """
        self.screen = Screen(
            width_px=screen_width_px,
            height_px=screen_height_px,
            width_cm=screen_width_cm,
            height_cm=screen_height_cm,
            distance_cm=distance_cm,
        )
        self.sampling_rate = sampling_rate
