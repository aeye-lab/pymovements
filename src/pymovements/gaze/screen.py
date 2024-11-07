# Copyright (c) 2022-2024 The pymovements Project Authors
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
"""Provides the Screen class."""
from __future__ import annotations

import numpy as np

from pymovements.gaze import transforms_numpy
from pymovements.utils import checks
from pymovements.utils import decorators


@decorators.auto_str
class Screen:
    """Screen class for holding screen properties.

     Also transforms pixel coordinates to degrees of visual angle.

    Parameters
    ----------
    width_px: int | None
        Screen width in pixels. (default: None)
    height_px: int | None
        Screen height in pixels. (default: None)
    width_cm: float | None
        Screen width in centimeters. (default: None)
    height_cm: float | None
        Screen height in centimeters. (default: None)
    distance_cm: float | None
        Eye-to-screen distance in centimeters. If None, a `distance_column` must be provided
        in the `DatasetDefinition` or `GazeDataFrame`, which contains the eye-to-screen
        distance for each sample in millimeters. (default: None)
    origin: str | None
        Specifies the screen location of the origin of the pixel
        coordinate system. (default: 'upper left')

    Examples
    --------
    >>> screen = Screen(
    ...     width_px=1280,
    ...     height_px=1024,
    ...     width_cm=38.0,
    ...     height_cm=30.0,
    ...     distance_cm=68.0,
    ...     origin='upper left',
    ... )
    >>> print(screen)
    Screen(width_px=1280, height_px=1024, width_cm=38.00,
    height_cm=30.00, distance_cm=68.00, origin=upper left)

    We can also access the screen boundaries in degrees of visual angle. This only works if the
    `distance_cm` attribute is specified.

    >>> screen.x_min_dva# doctest:+ELLIPSIS
    -15.59...
    >>> screen.x_max_dva# doctest:+ELLIPSIS
    15.59...
    >>> screen.y_min_dva# doctest:+ELLIPSIS
    -12.42...
    >>> screen.y_max_dva# doctest:+ELLIPSIS
    12.42...

    """

    def __init__(
            self,
            width_px: int | None = None,
            height_px: int | None = None,
            width_cm: float | None = None,
            height_cm: float | None = None,
            distance_cm: float | None = None,
            origin: str | None = 'upper left',
    ):
        if width_px is not None:
            checks.check_is_greater_than_zero(width_px=width_px)
        self.width_px = width_px

        if height_px is not None:
            checks.check_is_greater_than_zero(height_px=height_px)
        self.height_px = height_px

        if width_cm is not None:
            checks.check_is_greater_than_zero(width_cm=width_cm)
        self.width_cm = width_cm

        if height_cm is not None:
            checks.check_is_greater_than_zero(height_cm=height_cm)
        self.height_cm = height_cm

        if distance_cm is not None:
            checks.check_is_greater_than_zero(distance_cm=distance_cm)
        self.distance_cm = distance_cm

        self.origin = origin

    @property
    def x_max_dva(self) -> float:
        """Maximum screen x-coordinate in degrees of visual angle."""
        self._check_numerical_attribute('width_px')
        assert self.width_px is not None

        self._check_numerical_attribute('width_cm')
        assert self.width_cm is not None

        self._check_numerical_attribute('distance_cm')
        assert self.distance_cm is not None

        checks.check_is_not_none(origin=self.origin)
        assert self.origin is not None

        return float(
            transforms_numpy.pix2deg(
                self.width_px - 1,
                screen_px=self.width_px,
                screen_cm=self.width_cm,
                distance_cm=self.distance_cm,
                origin=self.origin,
            ),
        )

    @property
    def y_max_dva(self) -> float:
        """Maximum screen y-coordinate in degrees of visual angle."""
        self._check_numerical_attribute('height_px')
        assert self.height_px is not None

        self._check_numerical_attribute('height_cm')
        assert self.height_cm is not None

        self._check_numerical_attribute('distance_cm')
        assert self.distance_cm is not None

        checks.check_is_not_none(origin=self.origin)
        assert self.origin is not None

        return float(
            transforms_numpy.pix2deg(
                self.height_px - 1,
                screen_px=self.height_px,
                screen_cm=self.height_cm,
                distance_cm=self.distance_cm,
                origin=self.origin,
            ),
        )

    @property
    def x_min_dva(self) -> float:
        """Minimum screen x-coordinate in degrees of visual angle."""
        self._check_numerical_attribute('width_px')
        assert self.width_px is not None

        self._check_numerical_attribute('width_cm')
        assert self.width_cm is not None

        self._check_numerical_attribute('distance_cm')
        assert self.distance_cm is not None

        checks.check_is_not_none(origin=self.origin)
        assert self.origin is not None

        return float(
            transforms_numpy.pix2deg(
                0,
                screen_px=self.width_px,
                screen_cm=self.width_cm,
                distance_cm=self.distance_cm,
                origin=self.origin,
            ),
        )

    @property
    def y_min_dva(self) -> float:
        """Minimum screen y-coordinate in degrees of visual angle."""
        self._check_numerical_attribute('height_px')
        assert self.height_px is not None

        self._check_numerical_attribute('height_cm')
        assert self.height_cm is not None

        self._check_numerical_attribute('distance_cm')
        assert self.distance_cm is not None

        checks.check_is_not_none(origin=self.origin)
        assert self.origin is not None

        return float(
            transforms_numpy.pix2deg(
                0,
                screen_px=self.height_px,
                screen_cm=self.height_cm,
                distance_cm=self.distance_cm,
                origin=self.origin,
            ),
        )

    def pix2deg(
            self,
            arr: float | list[float] | list[list[float]] | np.ndarray,
    ) -> np.ndarray:
        """Convert pixel screen coordinates to degrees of visual angle.

        Parameters
        ----------
        arr: float | list[float] | list[list[float]] | np.ndarray
            Pixel coordinates to transform into degrees of visual angle

        Returns
        -------
        np.ndarray
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
        ...     origin='upper left',
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
        self._check_numerical_attribute('width_px')
        assert self.width_px is not None

        self._check_numerical_attribute('height_px')
        assert self.height_px is not None

        self._check_numerical_attribute('width_cm')
        assert self.width_cm is not None

        self._check_numerical_attribute('height_cm')
        assert self.height_cm is not None

        self._check_numerical_attribute('distance_cm')
        assert self.distance_cm is not None

        checks.check_is_not_none(origin=self.origin)
        assert self.origin is not None

        return transforms_numpy.pix2deg(
            arr=arr,
            screen_px=(self.width_px, self.height_px),
            screen_cm=(self.width_cm, self.height_cm),
            distance_cm=self.distance_cm,
            origin=self.origin,
        )

    def _check_numerical_attribute(self, key: str) -> None:
        """Check if numerical attribute is not None and greater than zero."""
        value = getattr(self, key, None)
        checks.check_is_not_none(**{key: value})
        assert isinstance(value, (int, float))
        checks.check_is_greater_than_zero(**{key: value})
