"""
This module holds the synthetic eye gaze step function.
"""
from __future__ import annotations

from collections.abc import Sized

import numpy as np


def step_function(
    length: int,
    steps: list[int],
    values: list[float | tuple[float, ...]],
    start_value: float | tuple[float, ...] = 0,
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


    Examples
    --------
        >>> step_function(
        ...     length=10,
        ...     steps=[2, 5, 9],
        ...     values=[1, 2, 3],
        ...     start_value=0,
        ... )
        array([0., 0., 1., 1., 1., 2., 2., 2., 2., 3.])
    """
    # Check that steps and values have equal length.
    if len(steps) != len(values):
        raise ValueError(
            'length of steps not equal to length of values'
            f' ({len(steps)} != {len(values)})',
        )

    # Check that steps are sorted in ascending order.
    if sorted(steps) != steps:
        raise ValueError('steps must be sorted in ascending order.')

    # Infer number of channels from values.
    if isinstance(values[0], (int, float)):
        n_channels = 1
    else:
        n_channels = len(values[0])

    # Check that all values have equal number of channels.
    if n_channels == 1:
        if any(not isinstance(value, (int, float)) for value in values):
            raise ValueError('all values must have equal number of channels.')
    elif any(not isinstance(value, Sized) or len(value) != n_channels for value in values):
        raise ValueError('all values must have equal number of channels.')

    # Make sure start value corresponds to number of channels.
    if n_channels > 1:

        # If start value is a scalar, create tuple with length of number of channels.
        if isinstance(start_value, (int, float)):
            start_value = tuple(start_value for _ in range(n_channels))

        # Raise error if length of start value doesn't match n_channels.
        elif len(start_value) != n_channels:
            raise ValueError(
                'start_value must be scalar or must have same number of channels as values.',
            )

    # Initialize output array with start value.
    if n_channels == 1:
        arr = np.ones(length) * start_value
    else:
        arr = np.tile(start_value, (length, 1))

    # Iterate through all steps except the last.
    for begin, end, value in zip(steps[:-1], steps[1:], values[:-1]):
        arr[begin:end] = value

    # Set value for each step until the end.
    arr[steps[-1]:] = values[-1]

    return arr
