"""
This module holds the implementation of the ivt algorithm.
"""
from __future__ import annotations

import numpy as np

from pymovements.events import Fixation
from pymovements.transforms import norm


def ivt(
        positions: list[list[float]] | np.ndarray,
        velocities: list[list[float]] | np.ndarray,
        velocity_threshold: float,
) -> list[Fixation]:
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

    Returns
    -------
    list[Fixation]:
        List of Fixation events

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
            f"shape of velocities {velocities.shape}",
        )

    # Check if threshold is None
    if velocity_threshold is None:
        raise ValueError('velocity threshold is None')

    # Check if threshold is greater 0
    if velocity_threshold <= 0:
        raise ValueError('velocity threshold must be greater than 0')

    velocity_norm = norm(velocities, axis=1)

    # Map velocities lower than threshold to True and greater equals to False
    fix_map = velocity_norm < velocity_threshold

    # Find onsets for group of velocities
    loc_group_onsets = np.empty(len(positions), dtype=bool)
    loc_group_onsets[0] = True
    np.not_equal(fix_map[:-1], fix_map[1:], out=loc_group_onsets[1:])
    ind_group_onsets = np.nonzero(loc_group_onsets)[0]

    # Find offsets for group of velocities
    group_lengths = np.diff(np.append(ind_group_onsets, len(positions)))
    ind_group_offsets = np.add(ind_group_onsets, group_lengths)

    # Stack onsets and offsets and filter out fixation groups
    groups = np.stack((ind_group_onsets, ind_group_offsets), axis=1)
    fix_groups = groups[[fix_map[group[0]] == 1 for group in groups]]

    fixations = []

    for onset, offset in fix_groups:
        fixation_points = positions[onset:offset]
        centroid = np.mean(fixation_points, axis=0, dtype=np.float64)
        centroid = (centroid[0], centroid[1])

        fixations.append(Fixation(onset, offset, centroid))

    return fixations
