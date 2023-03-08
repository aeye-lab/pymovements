# Copyright (c) 2022-2023 The pymovements Project Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""This module holds the Screen class."""
from __future__ import annotations

import numpy as np

from pymovements.gaze import transforms
from pymovements.utils import checks
from pymovements.utils import decorators


@decorators.auto_str
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
        ...     width_cm=38.0,
        ...     height_cm=30.0,
        ...     distance_cm=68.0,
        ...     origin='lower left',
        ... )
        >>> print(screen)
        Screen(width_px=1280, height_px=1024, width_cm=38.00,
        height_cm=30.00, distance_cm=68.00, origin=lower left,
        x_max_dva=15.60, y_max_dva=12.43, x_min_dva=-15.60,
        y_min_dva=-12.43)

        """
        checks.check_no_zeros(width_px, 'width_px')
        checks.check_no_zeros(height_px, 'height_px')
        checks.check_no_zeros(width_cm, 'width_cm')
        checks.check_no_zeros(height_cm, 'height_cm')
        checks.check_no_zeros(distance_cm, 'distance_cm')

        self.width_px = width_px
        self.height_px = height_px
        self.width_cm = width_cm
        self.height_cm = height_cm
        self.distance_cm = distance_cm
        self.origin = origin

        # calculate screen boundary coordinates in degrees of visual angle
        self.x_max_dva = transforms.pix2deg(
            width_px - 1,
            screen_px=width_px, screen_cm=width_cm, distance_cm=distance_cm, origin=origin,
        )
        self.y_max_dva = transforms.pix2deg(
            height_px - 1,
            screen_px=height_px, screen_cm=height_cm, distance_cm=distance_cm, origin=origin,
        )
        self.x_min_dva = transforms.pix2deg(
            0,
            screen_px=width_px, screen_cm=width_cm, distance_cm=distance_cm, origin=origin,
        )
        self.y_min_dva = transforms.pix2deg(
            0,
            screen_px=height_px, screen_cm=height_cm, distance_cm=distance_cm, origin=origin,
        )

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
        >>> arr = [(123.0, 865.0)]
        >>> screen = Screen(
        ...     width_px=1280,
        ...     height_px=1024,
        ...     width_cm=38.0,
        ...     height_cm=30.0,
        ...     distance_cm=68.0,
        ...     origin='lower left',
        ... )
        >>> screen.pix2deg(arr=arr)
        array([[-12.70732231, 8.65963972]])

        >>> screen = Screen(
        ...     width_px=1280,
        ...     height_px=1024,
        ...     width_cm=38.0,
        ...     height_cm=30.0,
        ...     distance_cm=68.0,
        ...     origin='center',
        ... )
        >>> screen.pix2deg(arr=arr)
        array([[ 3.07379946, 20.43909054]])
        """
        return transforms.pix2deg(
            arr=arr,
            screen_px=(self.width_px, self.height_px),
            screen_cm=(self.width_cm, self.height_cm),
            distance_cm=self.distance_cm,
            origin=self.origin,
        )
