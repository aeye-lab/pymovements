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
This module holds the implementation for idt algorithm.
"""
from __future__ import annotations

import numpy as np

from pymovements.events.events import EventDataFrame
from pymovements.utils import checks
from pymovements.utils.filters import events_split_nans
from pymovements.utils.filters import filter_candidates_remove_nans


def dispersion(positions: list[list[float]] | np.ndarray) -> float:
    """
    Compute the dispersion of a group of consecutive points in a 2D position time series.

    The dispersion is defined as the sum of the differences between
    the points' maximum and minimum x and y values

    Parameters
    ----------
    positions: array-like
        Continuous 2D position time series.

    Returns
    -------
    dispersion: float
        Dispersion of the group of points.
    """
    return sum(np.nanmax(positions, axis=0) - np.nanmin(positions, axis=0))


def idt(
        positions: list[list[float]] | list[tuple[float, float]] | np.ndarray,
        velocities: list[list[float]] | list[tuple[float, float]] | np.ndarray,
        timesteps: list[int] | np.ndarray | None = None,
        minimum_duration: int = 100,
        dispersion_threshold: float = 1.0,
        include_nan: bool = False,
) -> EventDataFrame:
    """
    Fixation identification based on dispersion threshold.

    The algorithm identifies fixations by grouping consecutive points
    within a maximum separation (dispersion) threshold and a minimum duration threshold.
    The algorithm uses a moving window to check the dispersion of the points in the window.
    If the dispersion is below the threshold, the window represents a fixation,
    and the window is expanded until the dispersion is above threshold.

    The implementation and its default parameter values are based on the description and pseudocode
    from Salvucci and Goldberg :cite:p:`SalvucciGoldberg2000`.

    Parameters
    ----------
    positions: array-like, shape (N, 2)
        Continuous 2D position time series
    velocities: array-like, shape (N, 2)
        Corresponding continuous 2D velocity time series.
    timesteps: array-like, shape (N, )
        Corresponding continuous 1D timestep time series. If None, sample based timesteps are
        assumed.
    minimum_duration: int
        Minimum fixation duration. The duration is specified in the units used in ``timesteps``.
         If ``timesteps`` is None, then ``minimum_duration`` is specified in numbers of samples.
    dispersion_threshold: float
        Threshold for dispersion for a group of consecutive samples to be identified as fixation
    include_nan: bool
        Indicator, whether we want to split events on missing/corrupt value (np.nan)

    Returns
    -------
    pl.DataFrame
        A dataframe with detected fixations as rows.

    Raises
    ------
    ValueError
        If positions is not shaped (N, 2)
        If dispersion_threshold is not greater than 0
        If duration_threshold is not greater than 0
    """
    positions = np.array(positions)
    velocities = np.array(velocities)
    checks.check_shapes_positions_velocities(positions=positions, velocities=velocities)

    if timesteps is None:
        timesteps = np.arange(len(velocities), dtype=np.int64)
    timesteps = np.array(timesteps)
    checks.check_is_length_matching(velocities=velocities, timesteps=timesteps)

    if dispersion_threshold <= 0:
        raise ValueError('dispersion threshold must be greater than 0')
    if minimum_duration <= 0:
        raise ValueError('minimum duration must be greater than 0')

    onsets = []
    offsets = []

    # Infer minimum duration in number of samples.
    # This implementation is currently very restrictive.
    # It requires that the interval between timesteps is constant.
    # It requires that the minimum duration is divisible by the constant interval between timesteps.
    timesteps_diff = np.diff(timesteps)
    if not np.all(timesteps_diff == timesteps_diff[0]):
        raise ValueError('interval between timesteps must be constant')
    if not minimum_duration % timesteps_diff[0] == 0:
        raise ValueError(
            'minimum_duration must be divisible by the constant interval between timesteps',
        )
    minimum_sample_duration = minimum_duration // timesteps_diff[0]

    # Initialize window over first points to cover the duration threshold
    win_start = 0
    win_end = minimum_sample_duration

    while win_end < len(positions):

        # Initialize window over first points to cover the duration threshold.
        # This automatically extends the window to the specified minimum event duration.
        win_end = max(win_start + minimum_sample_duration, win_end)

        if dispersion(positions[win_start:win_end]) <= dispersion_threshold:
            # Add additional points to the window until dispersion > threshold.
            while dispersion(positions[win_start:win_end]) < dispersion_threshold:
                win_end += 1

                # break if we reach end of input data
                if win_end == len(positions):
                    break

            # check for np.nan values
            if np.sum(np.isnan(positions[win_start:win_end - 1])) > 0:
                tmp_candidates = [np.arange(win_start, win_end - 1, 1)]
                tmp_candidates = filter_candidates_remove_nans(
                    candidates=tmp_candidates,
                    values=positions,
                )
                # split events if include_nan == False
                if not include_nan:
                    tmp_candidates = events_split_nans(
                        candidates=tmp_candidates,
                        values=positions,
                    )

                # Filter all candidates by minimum duration.
                tmp_candidates = [
                    candidate for candidate in tmp_candidates
                    if len(candidate) >= minimum_sample_duration
                ]
                for candidate in tmp_candidates:
                    onsets.append(timesteps[candidate[0]])
                    offsets.append(timesteps[candidate[-1]])

            else:
                # Note a fixation at the centroid of the window points.

                onsets.append(timesteps[win_start])
                offsets.append(timesteps[win_end - 1])

            # Remove window points from points.
            # Initialize new window excluding the previous window
            win_start = win_end
        else:
            # Remove first point from points.
            # Move window start one step further without modifying window end.
            win_start += 1

    # Create proper flat numpy arrays.
    onsets_arr = np.array(onsets).flatten()
    offsets_arr = np.array(offsets).flatten()

    event_df = EventDataFrame(name='fixation', onsets=onsets_arr, offsets=offsets_arr)
    return event_df
