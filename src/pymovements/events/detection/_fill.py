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
"""This module holds the implementation of the event fill function."""
from __future__ import annotations

import numpy as np

from pymovements.events.detection._library import register_event_detection
from pymovements.events.frame import EventDataFrame
from pymovements.gaze.transforms_numpy import consecutive


@register_event_detection
def fill(
        events: EventDataFrame,
        timesteps: list[int] | np.ndarray,
        minimum_duration: int = 1,
        name: str = 'unclassified',
) -> EventDataFrame:
    """Classify all previously unclassified timesteps as events.

    Parameters
    ----------
    events:
        The already detected events.
    timesteps: array-like, shape (N, )
        Continuous 1D timestep time series.
    minimum_duration: int
        Minimum fixation duration. The duration is specified in the units used in ``timesteps``.
    name:
        Name for detected events in EventDataFrame.

    Returns
    -------
    pl.DataFrame
        A dataframe with detected fixations as rows.
    """
    timesteps = np.array(timesteps)

    # Create binary mask where each existing event is marked.
    events_mask = np.zeros(len(timesteps), dtype=bool)

    for row in events.frame.iter_rows(named=True):
        idx_onset = np.where(timesteps == row['onset'])[0][0]
        idx_offset = np.where(timesteps == row['offset'] - 1)[0][0]

        events_mask[idx_onset:idx_offset + 1] = True

    # Mask all indices where there is no event.
    candidate_mask = ~events_mask

    # Get indices of true values in candidate mask.
    candidate_indices = np.where(candidate_mask)[0]

    # Get all fixation candidates by grouping all consecutive indices.
    candidates = consecutive(arr=candidate_indices)

    if len(candidates) == 1 and np.array_equal(candidates[0], np.array([], dtype=np.int64)):
        return EventDataFrame()

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
    event_df = EventDataFrame(name=name, onsets=onsets, offsets=offsets)
    return event_df
