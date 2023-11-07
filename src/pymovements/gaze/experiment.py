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
"""This module holds the Experiment class."""
from __future__ import annotations

from typing import Any

import numpy as np

from pymovements.gaze import transforms_numpy
from pymovements.gaze.eyetracker import EyeTracker
from pymovements.gaze.screen import Screen
from pymovements.utils import checks


class Experiment:
    """Experiment class for holding experiment properties.

    Attributes
    ----------
    screen : Screen
        Screen object for experiment
    _sampling_rate : float
        Sampling rate in Hz
    eyetracker : EyeTracker | None
        Eye tracker for experiment
    """

    def __init__(
            self,
            screen_width_px: int,
            screen_height_px: int,
            screen_width_cm: float,
            screen_height_cm: float,
            distance_cm: float | None = None,
            origin: str = 'lower left',
            sampling_rate: float | None = None,
            eyetracker: EyeTracker | None = None,
    ):
        """Initialize Experiment.

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
        distance_cm : float | None
            Eye-to-screen distance in centimeters. If None, a `distance_column` must be provided
            in the `DatasetDefinition` or `GazeDataFrame`, which contains the eye-to-screen
            distance for each sample in millimeters.
        origin : str
            Specifies the screen location of the origin of the pixel coordinate system.
        sampling_rate : float
            Sampling rate in Hz
        eyetracker : EyeTracker
            EyeTracker object for experiment

        Examples
        --------
        >>> experiment = Experiment(
        ...     screen_width_px=1280,
        ...     screen_height_px=1024,
        ...     screen_width_cm=38,
        ...     screen_height_cm=30,
        ...     distance_cm=68,
        ...     origin='lower left',
        ...     sampling_rate=None,
        ...     eyetracker=EyeTracker(1000.0, False, True, 'EyeLink 1000 Plus', '1.5.3',
        ...                           'EyeLink', 'Arm Mount / Monocular / Remote',),
        ... )
        >>> print(experiment)
        Experiment(sampling_rate=1000.00, screen=Screen(width_px=1280, height_px=1024, width_cm=38,
        height_cm=30, distance_cm=68, origin=lower left),
        eyetracker=EyeTracker(sampling_rate=1000.0, left=False, right=True,
        model='EyeLink 1000 Plus', version='1.5.3', vendor='EyeLink',
        mount='Arm Mount / Monocular / Remote'), _sampling_rate=None)

        We can also access the screen boundaries in degrees of visual angle via the
        :py:attr:`~pymovements.gaze.Screen` object. This only works if the
        `distance_cm` attribute is specified.

        >>> experiment.screen.x_min_dva# doctest:+ELLIPSIS
        -15.59...
        >>> experiment.screen.x_max_dva# doctest:+ELLIPSIS
        15.59...
        >>> experiment.screen.y_min_dva# doctest:+ELLIPSIS
        -12.42...
        >>> experiment.screen.y_max_dva# doctest:+ELLIPSIS
        12.42...

        """
        self.screen = Screen(
            width_px=screen_width_px,
            height_px=screen_height_px,
            width_cm=screen_width_cm,
            height_cm=screen_height_cm,
            distance_cm=distance_cm,
            origin=origin,
        )

        checks.check_is_mutual_exclusive(sampling_rate=sampling_rate, eyetracker=eyetracker)

        self.eyetracker = eyetracker

        self._sampling_rate = sampling_rate

        checks.check_is_not_none(sampling_rate=self.sampling_rate)
        assert self.sampling_rate is not None

        checks.check_is_greater_than_zero(sampling_rate=self.sampling_rate)

    @property
    def sampling_rate(self) -> float:
        """Assign sampling rate to experiment."""
        if self._sampling_rate is not None:
            return self._sampling_rate
        if self.eyetracker is not None:
            return self.eyetracker.sampling_rate

    def pos2vel(
            self,
            arr: list[float] | list[list[float]] | np.ndarray,
            method: str = 'smooth',
            **kwargs: int | float | str,
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

        Examples
        --------
        >>> experiment = Experiment(
        ...     screen_width_px=1280,
        ...     screen_height_px=1024,
        ...     screen_width_cm=38,
        ...     screen_height_cm=30,
        ...     distance_cm=68,
        ...     origin='lower left',
        ...     sampling_rate=1000.0,
        ...     eyetracker=None,
        ... )
        >>> arr = [[0., 0.], [1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]]
        >>> experiment.pos2vel(
        ...    arr=arr,
        ...    method="smooth",
        ... )
        array([[ 500.,  500.],
               [1000., 1000.],
               [1000., 1000.],
               [1000., 1000.],
               [1000., 1000.],
               [ 500.,  500.]])
        """
        return transforms_numpy.pos2vel(
            arr=arr, sampling_rate=self.sampling_rate, method=method, **kwargs,
        )

    def __str__(self: Any) -> str:
        """Print experiment."""
        def shorten(value: Any) -> str:
            if isinstance(value, float):
                value = f'{value:.2f}'
            return value
        attributes = ', '.join(f'{key}={shorten(value)}' for key, value in vars(self).items())
        return f'{type(self).__name__}(sampling_rate={shorten(self.sampling_rate)}, {attributes})'
