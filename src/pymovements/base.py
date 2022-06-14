from __future__ import annotations

import numpy as np

from . import checks
from .transforms import pix2deg


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
    def __init__(self, width_px: int, height_px: int, width_cm: float,
                 height_cm: float, distance_cm: float):
        """
        Initializes Screen.

        Paramters
        ---------

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

        """
        checks.check_no_zeros(width_px, "width_px")
        checks.check_no_zeros(width_px, "height_px")
        checks.check_no_zeros(width_px, "width_cm")
        checks.check_no_zeros(width_px, "height_cm")
        checks.check_no_zeros(width_px, "distamce_cm")

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
            positions: float | list[float] | list[list[float]] | np.ndarray,
            center_origin: bool = True,
    ) -> np.ndarray:
        """
        Converts pixel screen coordinates to degrees of visual angle.

        Parameters
        ----------
        positions : float, array_like
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

        """
        return pix2deg(
            positions=positions,
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

    def __init__(self, screen_width_px: int, screen_height_px: int,
                 screen_width_cm: float, screen_height_cm: float,
                 distance_cm: float, sampling_rate: float):
        """
        Initializes Experiment.

        Paramters
        ---------

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

        """
        self.screen = Screen(
            width_px=screen_width_px,
            height_px=screen_height_px,
            width_cm=screen_width_cm,
            height_cm=screen_height_cm,
            distance_cm=distance_cm,
        )
        self.sampling_rate = sampling_rate
