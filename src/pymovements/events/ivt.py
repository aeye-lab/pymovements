from __future__ import annotations

from typing import Optional
import numpy as np

from pymovements.transforms import vnorm
from pymovements.events import Fixation


def ivt(
        positions: list[list[float]] | np.ndarray,
        velocities: Optional[list[list[float]] | np.ndarray],
        threshold: float
) -> list[Fixation]:
    """
    Identification of fixations based on velocity-threshold

    Parameters
    ----------
    positions: array-like
        Continuous 2D position time series.
    velocities: array-like
        Corresponding continuous 2D velocity time series.
    threshold: float
        Velocity threshold.

    Returns
    -------
    fixations: array
        List of fixations

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
    if not positions.shape == velocities.shape:
        raise ValueError(f"shape of positions {positions.shape} doesn't match"
                         f"shape of velocities {velocities.shape}")

    # Check if threshold is None
    if threshold is None:
        raise ValueError('velocity threshold is None')

    # Check if threshold is greater 0
    if threshold <= 0:
        raise ValueError('velocity threshold must be greater than 0')

    velocity_norm = vnorm(velocities, axis=1)

    # Map velocities lower than threshold to True and greater equals to False
    fix_map = velocity_norm < threshold

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
        centroid = tuple(np.mean(fixation_points, axis=0))

        fixations.append(Fixation(onset, offset, centroid))

    return fixations
