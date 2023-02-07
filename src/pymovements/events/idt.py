"""
This module holds the implementation for idt algorithm.
"""
from __future__ import annotations

import numpy as np

from pymovements.events import Fixation


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
    return sum(np.max(positions, axis=0) - np.min(positions, axis=0))


def idt(
        positions: list[list[float]] | np.ndarray,
        dispersion_threshold: float,
        duration_threshold: int,
) -> list[Fixation]:
    """
    Fixation identification based on dispersion threshold.

    The algorithm identifies fixations by grouping consecutive points
    within a maximum separation (dispersion) threshold and a minimum duration threshold.
    The algorithm uses a moving window to check the dispersion of the points in the window.
    If the dispersion is below the threshold, the window represents a fixation,
    and the window is expanded until the dispersion is above threshold.

    The implementation is based on the description and pseudocode
    from Salvucci and Goldberg :cite:p:`SalvucciGoldberg2000`.

    Parameters
    ----------
    positions: array-like, shape (N, 2)
        Continuous 2D position time series
    dispersion_threshold: float
        Threshold for dispersion for a group of consecutive samples to be identified as fixation
    duration_threshold: int
        Minimum fixation duration in number of samples

    Returns
    -------
    list[Fixation]:
        List of Fixation events

    Raises
    ------
    ValueError
        If positions is not shaped (N, 2)
        If dispersion_threshold is not greater than 0
        If duration_threshold is not greater than 0
    """

    positions = np.array(positions)

    # make sure positions is 2d
    if positions.ndim != 2:
        raise ValueError('positions needs to have shape (N, 2)')

    # make sure positions has shape (N, 2)
    if positions.shape[1] != 2:
        raise ValueError('positions needs to have shape (N, 2)')

    # Check if dispersion_threshold is greater 0
    if dispersion_threshold <= 0:
        raise ValueError('dispersion threshold must be greater than 0')

    # Check if duration_threshold is greater 0
    if duration_threshold <= 0:
        raise ValueError('duration threshold must be greater than 0')

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
            centroid = np.mean(positions[win_start:win_end], axis=0)
            centroid = (centroid[0], centroid[1])

            fixations.append(Fixation(win_start, win_end, centroid))

            # Initialize new window excluding the previous window
            win_start = win_end
            win_end = win_start + duration_threshold
        else:
            win_start += 1

    return fixations
