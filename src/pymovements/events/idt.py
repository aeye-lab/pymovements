from __future__ import annotations

import numpy as np

from pymovements.events import Fixation


def dispersion(positions):
    return np.sum(np.max(positions, axis=0) - np.min(positions, axis=0))


def idt(
        positions: list[list[float]] | np.ndarray,
        dispersion_threshold: float,
        duration_threshold: int,
) -> list[Fixation]:
    """
    Fixation identification based on dispersion threshold.

    The implementation is based on the description and pseudocode
    from Salvucci and Goldberg :cite:p:`SalvucciGoldberg2000`.

    Parameters
    ----------
    positions: array-like
        Continuous 2D position time series
    dispersion_threshold: float
        Threshold for dispersion for a group of consecutive samples to be identified as fixation
    duration_threshold: int
        Minimum fixation duration in number of samples

    Returns
    -------
    fixations:
        List of Fixation events

    Raises
    ------
    ValueError
        If positions is not shaped (N, 2)
        If dispersion_threshold is not greater than 0
        If duration_threshold is not greater than 0
    """

    positions = np.array(positions)

    # make sure x is 2d
    if positions.ndim != 2:
        raise ValueError(
            'positions needs to have shape (N, 2)'
        )

    # make sure x has shape (n, 2)
    if positions.shape[1] != 2:
        raise ValueError(
            'positions needs to have shape (N, 2)'
        )

    # Check if dispersion_threshold is greater 0
    if dispersion_threshold <= 0:
        raise ValueError(
            'dispersion threshold must be greater than 0'
        )

    # Check if duration_threshold is greater 0
    if duration_threshold <= 0:
        raise ValueError(
            'duration threshold must be greater than 0'
        )

    fixations = []

    # Initialize window over first points to cover the duration threshold
    win_start = 0
    win_end = duration_threshold

    while win_end < len(positions):
        if dispersion(positions[win_start:win_end]) <= dispersion_threshold:
            # Add additional points to the window until dispersion > threshold
            while dispersion(positions[win_start:win_end]) < dispersion_threshold:
                win_end += 1

                # break if we reach end of input data
                if win_end == len(positions) - 1:
                    break

            # Note a fixation at the centroid of the window points
            centroid = tuple(np.mean(positions[win_start:win_end], axis=0))
            fixations.append(Fixation(win_start, win_end, centroid))

            # Initialize new window excluding the previous window
            win_start = win_end
            win_end = win_start + duration_threshold
        else:
            win_start += 1

    return fixations
