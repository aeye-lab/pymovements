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

from pymovements.events.events import EventDataFrame
from pymovements.gaze.transforms import consecutive
from pymovements.gaze.transforms import norm
from pymovements.utils import checks
from pymovements.utils.filters import filter_candidates_remove_nans


def ivt(
        positions: list[list[float]] | list[tuple[float, float]] | np.ndarray,
        velocities: list[list[float]] | list[tuple[float, float]] | np.ndarray,
        timesteps: list[int] | np.ndarray | None = None,
        minimum_duration: int = 100,
        velocity_threshold: float = 20.0,
        include_nan: bool = False,
) -> EventDataFrame:
    """
    Identification of fixations based on velocity-threshold

    The algorithm classifies each point as a fixation if the velocity is below
    the given velocity threshold. Consecutive fixation points are merged into
    one fixation.

    The implementation and its default parameter values are based on the description and pseudocode
    from Salvucci and Goldberg :cite:p:`SalvucciGoldberg2000`.

    Parameters
    ----------
    positions: array-like, shape (N, 2)
        Continuous 2D position time series.
    velocities: array-like, shape (N, 2)
        Corresponding continuous 2D velocity time series.
    timesteps: array-like, shape (N, )
        Corresponding continuous 1D timestep time series. If None, sample based timesteps are
        assumed.
    minimum_duration: int
        Minimum fixation duration. The duration is specified in the units used in ``timesteps``.
         If ``timesteps`` is None, then ``minimum_duration`` is specified in numbers of samples.
    velocity_threshold: float
        Threshold for a point to be classified as a fixation. If the
        velocity is below the threshold, the point is classified as a fixation.
    include_nan: bool
        Indicator, whether we want to split events on missing/corrupt value (np.nan)

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

    checks.check_shapes_positions_velocities(positions=positions, velocities=velocities)
    if velocity_threshold is None:
        raise ValueError('velocity threshold must not be None')
    if velocity_threshold <= 0:
        raise ValueError('velocity threshold must be greater than 0')

    if timesteps is None:
        timesteps = np.arange(len(velocities), dtype=np.int64)
    timesteps = np.array(timesteps)
    checks.check_is_length_matching(velocities=velocities, timesteps=timesteps)

    # Get all indices with norm-velocities below threshold.
    velocity_norm = norm(velocities, axis=1)
    candidate_mask = velocity_norm < velocity_threshold

    # Add nans to candidates if desired.
    if include_nan:
        candidate_mask = np.logical_or(candidate_mask, np.isnan(velocities).any(axis=1))

    # Get indices of true values in candidate mask.
    candidate_indices = np.where(candidate_mask)[0]

    # Get all fixation candidates by grouping all consecutive indices.
    candidates = consecutive(arr=candidate_indices)

    # Remove leading and trailing nan values from candidates.
    if include_nan:
        candidates = filter_candidates_remove_nans(candidates=candidates, values=velocities)

    # Filter all candidates by minimum duration.
    candidates = [
        candidate for candidate in candidates
        if timesteps[candidate[-1]] - timesteps[candidate[0]] >= minimum_duration
    ]

    # Onset of each event candidate is first index in candidate indices.
    onsets = timesteps[[candidate_indices[0] for candidate_indices in candidates]].flatten()
    # Offset of each event candidate is last event in candidate indices.
    offsets = timesteps[[candidate_indices[-1] for candidate_indices in candidates]].flatten()

    # Create event dataframe from onsets and offsets.
    event_df = EventDataFrame(name='fixation', onsets=onsets, offsets=offsets)
    return event_df
