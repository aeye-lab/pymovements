"""
This module holds the implementation for the Engbert microsaccades algorithm.
"""


from __future__ import annotations

import numpy as np

from pymovements.events import Event


def microsaccades(
    positions: np.ndarray,
    velocities: np.ndarray,
    threshold: tuple[float] | None = None,
    threshold_factor: float = 6,
    threshold_method: str = 'engbert2015',
    min_duration: int = 6,
    min_threshold: float = 1e-10,
) -> list[Event]:
    """
    Compute (micro-)saccades from raw samples
    adopted from Engbert et al Microsaccade Toolbox 0.9

    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
        x und y coordinates of N samples in chronological order
    velocities : np.ndarray, shape (N, 2)
        x and y velocities of N samples in chronological order
    threshold : float
         if None: data-driven velocity threshold; if tuple of
        floats: used to compute elliptic threshold
    threshold_factor : float
        factor for relative velocity threshold computation
    threshold_method : str
        If no threshold given, compute it with this method.
    min_duration : int
        minimal saccade duration in samples
    min_threshold : float
        minimal threshold value. Raises ValueError if calculated threshold is too low.

    Returns
    -------
        list of arrays of shape (7,): (1) saccade onset, (2) saccade
            offset, (3) peak velocity, (4) horizontal component (dist from first
            to last sample of the saccade), (5) vertical component,
            (6) horizontal amplitude (dist from leftmost to rightmost sample),
            (7) vertical amplitude

    Raises
    ------
        ValueError
            If x.shape and v.shape do not match.
            If threshold value is below min_threshold value.
            If passed threshold is not two-dimensional.
    """
    positions = np.array(positions)
    velocities = np.array(velocities)

    if positions.shape != velocities.shape:
        raise ValueError('x.shape and v.shape do not match')

    if threshold is None:
        threshold = compute_sigma(velocities, method=threshold_method)
    else:
        if len(threshold) != 2:
            raise ValueError('threshold needs to be two-dimensional')
        threshold = np.array(threshold)

    if (threshold < min_threshold).any():
        raise ValueError(
            f'Threshold does not provide enough variance ({threshold} < {min_threshold})'
        )

    # radius of elliptic threshold
    radius = threshold * threshold_factor

    # test is <1 iff sample within ellipse
    test = np.power((velocities[:, 0] / radius[0]), 2) + np.power((velocities[:, 1] / radius[1]), 2)
    print(f'{test[765]:.100f}')
    print(f'{test[766]:.100f}')
    print(f'{test[767]:.100f}')
    # indices of candidate saccades
    # runtime warning because of nans in test
    # => is ok, the nans come from nans in x
    indx = np.where(np.greater(test, 1))[0]

    # Initialize saccade variables
    n = len(indx)  # number of saccade candidates
    nsac = 0
    saccades = []
    dur = 1
    a = 0  # (potential) saccade onset
    k = 0  # (potential) saccade offset, will be looped over
    issac = np.zeros(len(positions))  # codes if row in x is a saccade

    print(indx)

    # Loop over saccade candidates
    while k < n - 1:
        # check for ongoing saccade candidate and increase duration by one
        if indx[k + 1] - indx[k] == 1:
            dur += 1

        # else saccade has ended
        else:
            # check minimum duration criterion (exception: last saccade)
            if dur >= min_duration:
                nsac += 1
                s = np.zeros(7)  # entry for this saccade
                s[0] = indx[a]  # saccade onset
                s[1] = indx[k]  # saccade offset
                saccades.append(s)
                # code as saccade from onset to offset
                issac[indx[a]:indx[k]+1] = 1

            a = k + 1  # potential onset of next saccade
            dur = 1  # reset duration
        k += 1

    # Check minimum duration for last microsaccade
    if dur >= min_duration:
        nsac += 1
        s = np.zeros(7)  # entry for this saccade
        s[0] = indx[a]  # saccade onset
        s[1] = indx[k]  # saccade offset
        saccades.append(s)
        # code as saccade from onset to offset
        issac[indx[a]:indx[k]+1] = 1

    saccades = np.array(saccades)

    if nsac > 0:
        # Compute peak velocity, horizontal and vertical components
        for s in range(nsac):  # loop over saccades
            # Onset and offset for saccades
            a = int(saccades[s, 0])  # onset of saccade s
            b = int(saccades[s, 1])  # offset of saccade s
            idx = range(a, b+1)  # indices of samples belonging to saccade s
            print(list(idx))
            # Saccade peak velocity (vpeak)
            saccades[s, 2] = np.max(np.sqrt(
                np.power(velocities[idx, 0], 2) + np.power(velocities[idx, 1], 2)))
            # saccade length measured as distance between first (onset) and last (offset) sample
            saccades[s, 3] = positions[b, 0]-positions[a, 0]
            saccades[s, 4] = positions[b, 1]-positions[a, 1]
            # Saccade amplitude: saccade length measured as distance between
            # leftmost and rightmost (bzw. highest and lowest) sample
            minx = np.min(positions[idx, 0])  # smallest x-coordinate during saccade
            maxx = np.max(positions[idx, 0])
            miny = np.min(positions[idx, 1])
            maxy = np.max(positions[idx, 1])
            # direction of saccade; np.where returns tuple;
            # there could be more than one minimum/maximum => chose the first one
            signx = np.sign(np.where(
                positions[idx, 0] == maxx)[0][0] - np.where(positions[idx, 0] == minx)[0][0])
            signy = np.sign(np.where(
                positions[idx, 1] == maxy)[0][0] - np.where(positions[idx, 1] == miny)[0][0])
            saccades[s, 5] = signx * (maxx-minx)  # x-amplitude
            saccades[s, 6] = signy * (maxy-miny)  # y-amplitude

    return saccades


def compute_sigma(v: np.ndarray, method='engbert2015'):
    """
    Compute variation in velocity (sigma) by taking median-based std of x-velocity

    engbert2003:
    Ralf Engbert and Reinhold Kliegl: Microsaccades uncover the orientation of
    covert attention

    """
    if method == 'std':
        thx = np.nanstd(v[:, 0])
        thy = np.nanstd(v[:, 1])

    elif method == 'mad':
        thx = np.nanmedian(np.absolute(v[:, 0] - np.nanmedian(v[:, 0])))
        thy = np.nanmedian(np.absolute(v[:, 1] - np.nanmedian(v[:, 1])))

    elif method == 'engbert2003':
        thx = np.sqrt(np.nanmedian(np.power(v[:, 0], 2))
                      - np.power(np.nanmedian(v[:, 0]), 2))
        thy = np.sqrt(np.nanmedian(np.power(v[:, 1], 2))
                      - np.power(np.nanmedian(v[:, 1]), 2))

    elif method == 'engbert2015':
        thx = np.sqrt(np.nanmedian(np.power(v[:, 0] - np.nanmedian(v[:, 0]), 2)))
        thy = np.sqrt(np.nanmedian(np.power(v[:, 1] - np.nanmedian(v[:, 1]), 2)))

    else:
        valid_methods = ['std', 'mad', 'engbert2003', 'engbert2015']
        raise ValueError(
            f'Method "{method}" not implemented. Valid methods: {valid_methods}')

    return np.array([thx, thy])
