from __future__ import annotations

import numpy as np

from pymovements.events import Fixation


def dispersion(x):
    return np.sum(np.max(x, axis=0) - np.min(x, axis=0))


def idt(
        x: list[list[float]] | np.ndarray,
        dispersion_threshold: float,
        duration_threshold: int,
) -> list[Fixation]:
    """
    Fixation identification based on dispersion threshold.

    Parameters
    ----------
    x: array-like
        Continuous 2D position time series
    dispersion_threshold: float
        Threshold for dispersion for a group of consecutive points to be identified as fixation
    duration_threshold: int
        Minimum fixation duration

    Returns
    -------
    fixations:
        List of Fixation events
    """
    x = np.array(x)

    # make sure x has shape (n, 2)
    if x.ndim != 2 and x.shape[1] != 2:
        raise ValueError(
            'x needs to have shape (n, 2)'
        )

    # Check if dispersion_threshold is greater 0
    if not dispersion_threshold > 0:
        raise ValueError(
            'dispersion threshold must be greater than 0'
        )

    # Check if duration_threshold is greater 0
    if not duration_threshold > 0:
        raise ValueError(
            'duration threshold must be greater than 0'
        )

    fixations = []

    # Initialize window over first points to cover the duration threshold
    win_start = 0
    win_end = duration_threshold

    while win_end < len(x):
        if dispersion(x[win_start:win_end]) <= dispersion_threshold:
            # Add additional points to the window until dispersion > threshold
            while dispersion(x[win_start:win_end]) < dispersion_threshold:
                win_end += 1

                # break if we reach end of input data
                if win_end == len(x) - 1:
                    break

            # Note a fixation at the centroid of the window points
            centroid = np.sum(x[win_start:win_end], axis=0) / len(x[win_start:win_end])
            onset = win_start
            offset = win_end

            fixations.append(Fixation(onset, offset, centroid))

            # Initialize new window excluding the previous window
            win_start = win_end
            win_end = win_start + duration_threshold
        else:
            win_start += 1

    return fixations
