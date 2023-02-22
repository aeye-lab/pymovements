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
"""
This module holds the implementation for the Engbert microsaccades algorithm.
"""
from __future__ import annotations

from collections.abc import Sized

import numpy as np
import polars as pl

from pymovements.events.events import Saccade
from pymovements.transforms import consecutive
from pymovements.utils.checks import check_shapes_positions_velocities


def microsaccades(
    positions: list[list[float]] | list[tuple[float, float]] | np.ndarray,
    velocities: list[list[float]] | list[tuple[float, float]] | np.ndarray,
    threshold: np.ndarray | tuple[float] | str = 'engbert2015',
    threshold_factor: float = 6,
    minimum_duration: int = 6,
    minimum_threshold: float = 1e-10,
) -> pl.DataFrame:
    """Detect micro-saccades from velocity gaze sequence.

    This algorithm has a noise-adaptive velocity threshold parameter, which can also be set
    explicitly.

    The implemetation is based on the description from Engbert & Kliegl :cite:p:`EngbertKliegl2003`
    and is adopted from the Microsaccade Toolbox 0.9 originally implemented in R
    :cite:p:`Engbert2015`.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
        x and y positions of N samples in chronological order
    velocities : np.ndarray, shape (N, 2)
        x and y velocities of N samples in chronological order
    threshold : np.ndarray, tuple[float, float] or str
        If tuple of floats then use this as explicit elliptic threshold. If str, then use
        a data-driven velocity threshold method. See :func:`~events.engbert.compute_threshold` for
        a reference of valid methods. Default: `engbert2015`
    threshold_factor : float
        factor for relative velocity threshold computation. Default: 6
    minimum_duration : int
        minimal saccade duration in samples. Default: 6
    minimum_threshold : float
        minimal threshold value. Raises ValueError if calculated threshold is too low.
        Default: 1e-10

    Returns
    -------
    pl.DataFrame
        A dataframe with detected saccades as rows.

    Raises
    ------
    ValueError
        If `threshold` value is below `min_threshold` value.
        If passed `threshold` is either not two-dimensional or not a supported method.
    """
    positions = np.array(positions)
    velocities = np.array(velocities)

    check_shapes_positions_velocities(positions=positions, velocities=velocities)

    if isinstance(threshold, str):
        threshold = compute_threshold(velocities, method=threshold)
    else:
        if isinstance(threshold, Sized) and len(threshold) != 2:
            raise ValueError('threshold must be either string or two-dimensional')
        threshold = np.array(threshold)

    if (threshold < minimum_threshold).any():
        raise ValueError(
            'threshold does not provide enough variance as required by min_threshold'
            f' ({threshold} < {minimum_threshold})',
        )

    # Radius of elliptic threshold.
    radius = threshold * threshold_factor

    # If value is greater than 1, point lies outside the ellipse.
    outside_ellipse = np.greater(np.sum(np.power(velocities / radius, 2), axis=1), 1)

    # Get all indices with velocities outside of ellipse.
    outside_ellipse_indices = np.where(outside_ellipse)[0]

    # Get all saccade candidates by grouping all consecutive indices.
    candidates = consecutive(arr=outside_ellipse_indices)

    # Filter all candidates by minimum duration.
    candidates = [candidate for candidate in candidates if len(candidate) >= minimum_duration]

    # Create saccades from valid candidates. First channel is onset, second channel is offset.
    saccades = np.array([
        (candidate_indices[0], candidate_indices[-1])
        for candidate_indices in candidates
    ])

    if len(saccades) > 0:
        # Create event dataframe.
        event_df = pl.from_dict(
            {
                'type': 'saccade',
                'onset': saccades[:, 0].tolist(),
                'offset': saccades[:, 1].tolist(),
            },
            schema=Saccade.schema,
        )

    else:
        # Create empty dataframe with correct schema if no events detected.
        event_df = pl.DataFrame(schema=Saccade.schema)

    return event_df


def compute_threshold(arr: np.ndarray, method: str = 'engbert2015') -> np.ndarray:
    """Determine threshold by computing variation.

    The following methods are supported:

    - `std`: This is the channel-wise standard deviation.
    - `mad`: This is the channel-wise median absolute deviation.
    - `engbert2003`: This is the threshold method as described in :cite:p:`EngbertKliegl2003`.
    - `engbert2015`: This is the threshold method as described in :cite:p:`Engbert2015`.

    Parameters
    ----------
    arr : np.ndarray
        Array for which threshold is to be computed.
    method : str
        Method for threshold computation.

    Returns
    -------
    np.ndarray
        Threshold values for horizontal and vertical direction.

    Raises
    ------
    ValueError
        If passed method is not supported.
    """
    if method == 'std':
        thx = np.nanstd(arr[:, 0])
        thy = np.nanstd(arr[:, 1])

    elif method == 'mad':
        thx = np.nanmedian(np.absolute(arr[:, 0] - np.nanmedian(arr[:, 0])))
        thy = np.nanmedian(np.absolute(arr[:, 1] - np.nanmedian(arr[:, 1])))

    elif method == 'engbert2003':
        thx = np.sqrt(
            np.nanmedian(np.power(arr[:, 0], 2)) - np.power(np.nanmedian(arr[:, 0]), 2),
        )
        thy = np.sqrt(
            np.nanmedian(np.power(arr[:, 1], 2)) - np.power(np.nanmedian(arr[:, 1]), 2),
        )

    elif method == 'engbert2015':
        thx = np.sqrt(np.nanmedian(np.power(arr[:, 0] - np.nanmedian(arr[:, 0]), 2)))
        thy = np.sqrt(np.nanmedian(np.power(arr[:, 1] - np.nanmedian(arr[:, 1]), 2)))

    else:
        valid_methods = ['std', 'mad', 'engbert2003', 'engbert2015']
        raise ValueError(f'Method "{method}" not implemented. Valid methods: {valid_methods}')

    return np.array([thx, thy])
