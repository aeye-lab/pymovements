"""
Transforms module.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy

from pymovements.utils import checks


def pix2deg(
        arr: float | list[float] | list[list[float]] | np.ndarray,
        screen_px: float | list[float] | tuple[float, float] | np.ndarray,
        screen_cm: float | list[float] | tuple[float, float] | np.ndarray,
        distance_cm: float,
        origin: str,
) -> np.ndarray:
    """Converts pixel screen coordinates to degrees of visual angle.

    Parameters
    ----------
    arr : float, array_like
        Pixel coordinates to transform into degrees of visual angle
    screen_px : int, int
        Screen dimension in pixels
    screen_cm : float, float
        Screen dimension in centimeters
    distance_cm : float
        Eye-to-screen distance in centimeters
    origin : str
        Specifies the screen location of the origin of the pixel coordinate system. Valid values
        are: center, lower left.

    Returns
    -------
    np.ndarray
        Coordinates in degrees of visual angle

    Raises
    ------
    TypeError
        If arr is None.
    ValueError
        If dimension screen_px or screen_cm don't match dimension of arr.
        If screen_px or screen_cm or one of its elements is zero.
        If distance_cm is zero.
        If origin value is not supported.

    """
    if arr is None:
        raise TypeError("arr must not be None")

    checks.check_no_zeros(screen_px, "screen_px")
    checks.check_no_zeros(screen_cm, "screen_px")
    checks.check_no_zeros(distance_cm, "distance_cm")

    arr = np.array(arr)
    screen_px = np.array(screen_px)
    screen_cm = np.array(screen_cm)

    # Check basic arr dimensions.
    if arr.ndim not in [0, 1, 2]:
        raise ValueError(
            'Number of dimensions of arr must be either 0, 1 or 2'
            f' (arr.ndim: {arr.ndim})',
        )
    if arr.ndim == 2 and arr.shape[-1] not in [1, 2, 4]:
        raise ValueError(
            'Last coord dimension must have length 1, 2 or 4.'
            f' (arr.shape: {arr.shape})',
        )

    # check if arr dimensions match screen_px and screen_cm dimensions
    if arr.ndim in {0, 1}:
        if screen_px.ndim != 0 and screen_px.shape != (1,):
            raise ValueError('arr is 1-dimensional, but screen_px is not')
        if screen_cm.ndim != 0 and screen_cm.shape != (1,):
            raise ValueError('arr is 1-dimensional, but screen_cm is not')
    if arr.ndim != 0 and arr.shape[-1] == 2:
        if screen_px.shape != (2,):
            raise ValueError('arr is 2-dimensional, but screen_px is not')
        if screen_cm.shape != (2,):
            raise ValueError('arr is 2-dimensional, but screen_cm is not')
    if arr.ndim != 0 and arr.shape[-1] == 4:
        if screen_px.shape != (2,):
            raise ValueError('arr is 4-dimensional, but screen_px is not 2-dimensional')
        if screen_cm.shape != (2,):
            raise ValueError('arr is 4-dimensional, but screen_cm is not 2-dimensional')

        # We have binocular data. Double tile screen parameters.
        screen_px = np.tile(screen_px, 2)
        screen_cm = np.tile(screen_cm, 2)

    # Compute eye-to-screen-distance in pixels.
    distance_px = distance_cm * (screen_px / screen_cm)

    # If pixel coordinate system is not centered, shift pixel coordinate to the center.
    if origin == "lower left":
        arr = arr - (screen_px - 1) / 2
    elif origin != "center":
        raise ValueError(f"origin {origin} is not supported.")

    # 180 / pi transforms arc measure to degrees.
    return np.arctan2(arr, distance_px) * 180 / np.pi


def pos2vel(
        arr: list[float] | list[list[float]] | np.ndarray,
        sampling_rate: float = 1000,
        method: str = 'smooth',
        **kwargs,
) -> np.ndarray:
    """Compute velocity time series from 2-dimensional position time series.

    Methods 'smooth', 'neighbors' and 'preceding' are adapted from
        Engbert et al.: Microsaccade Toolbox 0.9.

    Parameters
    ----------
    arr : array_like
        Continuous 2D position time series
    sampling_rate : int
        Sampling rate of input time series
    method : str
        Following methods are available:
        * *smooth*: velocity is calculated from the difference of the mean values
        of the subsequent two samples and the preceding two samples
        * *neighbors*: velocity is calculated from difference of the subsequent
        sample and the preceding sample
        * *preceding*: velocity is calculated from the difference of the current
        sample to the preceding sample
    kwargs: dict
        Additional keyword arguments used for savitzky golay method.

    Returns
    -------
    np.ndarray
        Velocity time series in input_unit / sec

    Raises
    ------
    ValueError
        If selected method is invalid, input array is too short for the
        selected method or the sampling rate is below zero

    """
    if sampling_rate <= 0:
        raise ValueError('sampling_rate needs to be above zero')

    # make sure that we're operating on a numpy array
    arr = np.array(arr)

    if arr.ndim not in [1, 2]:
        raise ValueError(
            'arr needs to have 1 or 2 dimensions (are: {arr.ndim = })',
        )
    if method == 'smooth' and arr.shape[0] < 6:
        raise ValueError(
            'arr has to have at least 6 elements for method "smooth"',
        )
    if method == 'neighbors' and arr.shape[0] < 3:
        raise ValueError(
            'arr has to have at least 3 elements for method "neighbors"',
        )
    if method == 'preceding' and arr.shape[0] < 2:
        raise ValueError(
            'arr has to have at least 2 elements for method "preceding"',
        )
    if method != 'savitzky_golay' and kwargs:
        raise ValueError(
            'selected method doesn\'t support any additional kwargs',
        )

    N = arr.shape[0]
    v = np.zeros(arr.shape)

    valid_methods = ['smooth', 'neighbors', 'preceding', 'savitzky_golay']
    if method == 'smooth':
        # center is N - 2
        moving_avg = arr[4:N] + arr[3:N-1] - arr[1:N-3] - arr[0:N-4]
        # mean(arr_-2, arr_-1) and mean(arr_1, arr_2) needs division by two
        # window is now 3 samples long (arr_-1.5, arr_0, arr_1+5)
        # we therefore need a divison by three, all in all it's a division by 6
        v[2:N-2] = moving_avg * sampling_rate / 6

        # for second and second last sample:
        # calculate vocity from preceding and subsequent sample
        v[1] = (arr[2] - arr[0]) * sampling_rate / 2
        v[N-2] = (arr[N-1] - arr[N-3]) * sampling_rate / 2

        # for first and second sample:
        # calculate velocity from current and neighboring sample
        v[0] = (arr[1] - arr[0]) * sampling_rate / 2
        v[N-1] = (arr[N-1] - arr[N-2]) * sampling_rate / 2

    elif method == 'neighbors':
        # window size is two, so we need to divide by two
        v[1:N-1] = (arr[2:N] - arr[0:N-2]) * sampling_rate / 2

    elif method == 'preceding':
        v[1:N] = (arr[1:N] - arr[0:N-1]) * sampling_rate

    elif method == 'savitzky_golay':
        # transform to velocities
        if arr.ndim == 1:
            v = scipy.signal.savgol_filter(x=arr, **kwargs)
        else:  # we already checked for error cases

            for channel_id in range(arr.shape[1]):
                v[:, channel_id] = scipy.signal.savgol_filter(
                    x=arr[:, channel_id], **kwargs,
                )
        v = v * sampling_rate

    else:
        raise ValueError(
            f'Method needs to be in {valid_methods}'
            f' (is: {method})',
        )

    return v


def norm(arr: np.ndarray, axis: int | None = None) -> np.ndarray | Any:
    """
    Takes the norm sqrt(x^2 + y^2).

    Parameters
    ----------
    arr: np.ndarray
        velocity sequence
    axis: int, optional
        axis to take norm. If None it is inferred from arr.shape.

    Returns
    -------
    np.ndarray
    """
    if axis is None:
        # for single vector and array of vectors the axis is 0
        # shape is assumed to be either (2, ) or (2, sequence_length)
        if arr.ndim in {1, 2}:
            axis = 0

        # for batched array of vectors, the
        # shape is assumed to be (n_batches, 2, sequence_length)
        elif arr.ndim == 3:
            axis = 1

    return np.linalg.norm(arr, axis=axis)


def cut_into_subsequences(
    arr: np.ndarray, window_size: int, keep_padded: bool = True,
) -> np.ndarray:
    """
    Example: if old seq len was 7700, window_size=1000:
    Input arr has: 144 x 7700 x n_channels
    Output arr has: 144*8 x 1000 x n_channels
    The last piece of each trial 7000-7700 gets padded with first 300 of this piece to be 1000 long

    Parameters
    ----------
    arr: np.ndarray
        uncut sequence
    window_size: int
        size of subsequences
    keep_padded: bool
        If True, last subsequence (which is padded) is kept in the output array.

    Returns
    -------
    np.ndarray
    """
    n, rest = np.divmod(arr.shape[1], window_size)

    if rest > 0 and keep_padded:
        n_rows = arr.shape[0]*(n+1)
    else:
        n_rows = arr.shape[0]*n

    arr_cut = np.nan * np.ones((n_rows, window_size, arr.shape[2]))

    idx = 0
    for t in range(0, arr.shape[0]):
        for i in range(0, n):
            # cut out 1000 ms piece of trial t
            arr_tmp = np.expand_dims(arr[t, i*window_size: (i+1)*window_size, :], axis=0)

            # concatenate pieces
            arr_cut[idx, :, :] = arr_tmp

            idx = idx + 1

        if rest > 0 and keep_padded:
            # concatenate last one with pad
            start_idx_last_piece = window_size * n
            len_pad_to_add = window_size-rest
            # piece to pad:
            arr_incomplete = np.expand_dims(arr[t, start_idx_last_piece:arr.shape[1], :], axis=0)
            # padding piece:
            start_idx_last_piece = window_size * (n-1)
            arr_pad = np.expand_dims(
                arr[t, start_idx_last_piece:start_idx_last_piece+len_pad_to_add, :], axis=0,
            )

            arr_tmp = np.concatenate((arr_incomplete, arr_pad), axis=1)

            # concatenate last piece of original row t
            arr_cut[idx, :, :] = arr_tmp

            idx = idx + 1

    return arr_cut


def downsample(
        arr: np.ndarray,
        factor: int,
) -> np.ndarray:
    """
    Downsamples array by integer factor.

    Parameters
    ----------
    arr: np.ndarray
        sequence to be downsampled
    factor: int
        factor to be downsampled with

    Returns
    -------
    np.ndarray
    """
    sequence_length = arr.shape[0]
    select = [i % factor == 0 for i in range(sequence_length)]

    return arr[select].copy()


def consecutive(arr: np.ndarray) -> list[np.ndarray]:
    """Split array into groups of consecutive numbers.

    Parameters
    ----------
    arr : np.ndarray
        Array to be split into groups of consecutive numbers.

    Returns
    -------
    list[np.ndarray]
        List of arrays with consecutive numbers.

    Example
    -------
    >>> arr = np.array([0, 47, 48, 49, 50, 97, 98, 99])
    >>> consecutive(arr)
    [array([0]), array([47, 48, 49, 50]), array([97, 98, 99])]
    """
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)
