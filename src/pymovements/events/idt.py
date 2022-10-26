from __future__ import annotations

import numpy as np


def dispersion(x):
    return np.sum(np.max(x, axis=0), np.min(x, axis=0))


def idt(
        x: list[list[float]] | np.ndarray,
        dispersion_threshold: float,  # 1/2° to 1°
        duration_threshold: float  # 100 - 200 ms
):
    # TODO: Documentation and error checks

    x = np.array(x)

    centroids = []
    onsets = []
    offsets = []

    # Initialize window over first points to cover the duration threshold
    win_start = 0
    win_end = duration_threshold

    while win_end < len(x):
        if dispersion(x[win_start:win_end]) <= dispersion_threshold:
            # Add additional points to the window until dispersion > threshold
            while not dispersion(x[win_start:win_end]) > dispersion_threshold:
                win_end += 1

            # Note a fixation at the centroid of the window points
            centroids.append(np.sum(x[win_start:win_end], axis=0) / len(x[win_start:win_end]))
            onsets.append(win_start)
            offsets.append(win_end)

            # Initialize new window excluding the previous window
            win_start = win_end
            win_end = win_start + duration_threshold
        else:
            win_start += 1

    # TODO: return events
