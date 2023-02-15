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
This module holds the implementation of the ivt algorithm.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pymovements.transforms import consecutive
from pymovements.transforms import norm


def ivt(
        positions: list[list[float]] | np.ndarray,
        velocities: list[list[float]] | np.ndarray,
        velocity_threshold: float,
        minimum_duration: int,
) -> pl.DataFrame:
    """
    Identification of fixations based on velocity-threshold

    The algorithm classifies each point as a fixation if the velocity is below
    the given velocity threshold. Consecutive fixation points are merged into
    one fixation.

    The implementation is based on the description and pseudocode
    from Salvucci and Goldberg :cite:p:`SalvucciGoldberg2000`.

    Parameters
    ----------
    positions: array-like, shape (N, 2)
        Continuous 2D position time series.
    velocities: array-like, shape (N, 2)
        Corresponding continuous 2D velocity time series.
    velocity_threshold: float
        Threshold for a point to be classified as a fixation. If the
        velocity is below the threshold, the point is classified as a fixation.
    minimum_duration: int
        Minimum fixation duration in number of samples

    Returns
    -------
    pl.DataFrame
        A dataframe with detected fixations as rows.

    Raises
    ------
    ValueError
        If positions or velocities are None
        If positions or velocities do not have shape (N, 2)
        If positions and velocities have different shapes
        If velocity threshold is None.
        If velocity threshold is not greater than 0.
    """
    positions = np.array(positions)
    velocities = np.array(velocities)

    # make sure positions and velocities have shape (n, 2)
    if positions.ndim != 2:
        raise ValueError('positions need to have shape (N, 2)')

    if positions.shape[1] != 2:
        raise ValueError('positions need to have shape (N, 2)')

    if velocities.ndim != 2:
        raise ValueError('velocities need to have shape (N, 2)')

    if velocities.shape[1] != 2:
        raise ValueError('velocities need to have shape (N, 2)')

    # Check matching shape for positions and velocities
    if positions.shape != velocities.shape:
        raise ValueError(
            f"shape of positions {positions.shape} doesn't match"
            f'shape of velocities {velocities.shape}',
        )

    # Check if threshold is None
    if velocity_threshold is None:
        raise ValueError('velocity threshold is None')

    # Check if threshold is greater 0
    if velocity_threshold <= 0:
        raise ValueError('velocity threshold must be greater than 0')

    velocity_norm = norm(velocities, axis=1)

    # Get all indices with velocities outside of ellipse.
    below_threshold_indices = np.where(velocity_norm < velocity_threshold)[0]

    # Get all fixation candidates by grouping all consecutive indices.
    candidates = consecutive(arr=below_threshold_indices)

    # Filter all candidates by minimum duration.
    candidates = [candidate for candidate in candidates if len(candidate) >= minimum_duration]

    # Create ficaitons from valid candidates. First channel is onset, second channel is offset.
    fixations = np.array([
        (candidate_indices[0], candidate_indices[-1])
        for candidate_indices in candidates
    ])

    # Calculate centroid positions for fixations.
    centroids = [
        np.mean(positions[fixation[0]:fixation[1]], axis=0, dtype=np.float64).tolist()
        for fixation in fixations
    ]

    event_df = pl.from_dict({
        'type': 'fixation',
        'onset': fixations[:, 0].tolist(),
        'offset': fixations[:, 1].tolist(),
        'position': centroids,
    })
    return event_df
