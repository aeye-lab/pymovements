from __future__ import annotations

from typing import Optional
import numpy as np

from pymovements.transforms import vnorm
from pymovements.events import Fixation


def ivt(
        x: list[list[float]] | np.ndarray,
        v: Optional[list[list[float]] | np.ndarray],
        t: float
) -> list[Fixation]:
    """
    Identification of fixations based on velocity-threshold

    Parameters
    ----------
    x: array-like
        Continuous 2D position time series.
    v: array-like
        Corresponding continuous 2D velocity time series.
    t: float
        Velocity threshold.

    Returns
    -------
    fixations: array
        List of fixations

    """
    x = np.array(x)
    v = np.array(v)

    # Check if threshold is greater 0
    if not t > 0:
        raise ValueError(
            'velocity threshold must be greater than 0'
        )

    # make sure x and v have shape (n, 2)
    if x.ndim != 2 and x.shape[1] != 2:
        raise ValueError(
            'x needs to have shape (n, 2)'
        )

    if x.ndim != 2 and x.shape[1] != 2:
        raise ValueError(
            'v needs to have shape (n, 2)'
        )

    # Check matching shape for x and v
    if not x.shape == v.shape:
        raise ValueError(
            "shape x {} doesn't match shape v {}".format(x.shape, v.shape)
        )

    total_velocity = vnorm(v)

    # Map velocities lower than threshold to 1 and greater equals to 0
    fix_map = np.array(list(map(lambda point: 1 if point < t else 0, total_velocity)))

    # Find onsets for group of velocities
    loc_group_onsets = np.empty(len(x), dtype=bool)
    loc_group_onsets[0] = True
    np.not_equal(fix_map[:-1], fix_map[1:], out=loc_group_onsets[1:])
    ind_group_onsets = np.nonzero(loc_group_onsets)[0]

    # Find offsets for group of velocities
    group_lengths = np.diff(np.append(ind_group_onsets, len(x)))
    ind_group_offsets = np.add(ind_group_onsets, group_lengths)

    # Stack onsets and offsets and filter out fixation groups
    groups = np.stack((ind_group_onsets, ind_group_offsets), axis=1)
    fix_groups = groups[[fix_map[group[0]] == 1 for group in groups]]

    fixations = []

    for onset, offset in fix_groups:
        fixation_points = x[onset:offset]
        centroid = np.sum(fixation_points, axis=0) / len(fixation_points)

        fixations.append(Fixation(onset, offset, centroid))

    return fixations
