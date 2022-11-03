"""
This module holds the synthetic eye gaze step function.
"""


from __future__ import annotations

import numpy as np


def step_function(
    length: int,
    steps: list[int],
    values: list[float],
    start_value: float = 0,
) -> np.ndarray:
    """
    Create a synthetic eye gaze by using a simple step function.

    Parameters
    ----------
    length : int
        Length of the output sequence.
    steps : list[int]
        Indices for each step to happen.
    values : list[float]
        Array values to set at each step.
    start_value: int
        Array value to start with.

    Returns
    -------
    arr : np.ndarray
        Output signal

    Raises
    ------
    ValueError
        If steps not sorted in ascending order or length of steps not equal to length of values.

    """
    if len(steps) != len(values):
        raise ValueError('length of steps not equal to length of values'
                         f' ({len(steps)} != {len(values)})')

    if sorted(steps) != steps:
        raise ValueError('steps need to be sorted in ascending order.')

    arr = np.ones(length) * start_value

    # Iterate through all steps except the last.
    for begin, end, value in zip(steps[:-1], steps[1:], values[:-1]):
        arr[begin:end] = value

    # Set value for each step until the end.
    arr[steps[-1]:] = values[-1]

    return arr
