from __future__ import annotations

import numpy as np

from . import checks


def pix2deg(
        positions: float | list[float] | list[list[float]] | np.ndarray,
        screen_px: float | list[float] | tuple[float, float] | np.ndarray,
        screen_cm: float | list[float] | tuple[float, float] | np.ndarray,
        distance_cm: float,
        center_origin: bool = True,
) -> np.ndarray:
    """Converts pixel screen coordinates to degrees of visual angle.

    Parameters
    ----------
    positions : float, array_like
        Pixel coordinates to transform into degrees of visual angle
    screen_px : int, int
        Screen dimension in pixels
    screen_cm : float, float
        Screen dimension in centimeters
    distance_cm : float
        Eye-to-screen distance in centimeters
    center_origin: bool
        Center origin to (0,0) if positions origin is in bottom left corner

    Returns
    -------
    degrees_of_visual_angle : np.ndarray
        Coordinates in degrees of visual angle

    Raises
    ------
    ValueError
        If dimension screen_px or screen_cm don't match dimension of positions.
        If screen_px or screen_cm or one of its elements is zero.
        If distance_cm is zero.

    """
    checks.check_no_zeros(screen_px, "screen_px")
    checks.check_no_zeros(screen_cm, "screen_px")
    checks.check_no_zeros(distance_cm, "distance_cm")

    positions = np.array(positions)
    screen_px = np.array(screen_px)
    screen_cm = np.array(screen_cm)

    # check basic positions dimensions
    if positions.ndim not in [0, 1, 2]:
        raise ValueError(
            'Number of dimensions of positions must be either 0, 1 or 2'
            f' (positions.ndim: {positions.ndim})'
        )
    if positions.ndim == 2 and positions.shape[-1] > 2:
        raise ValueError(
            'Last coord dimension must have length 1 or 2.'
            f' (positions.shape: {positions.shape})'
        )

    # check if positions dimensions match screen_px and screen_cm dimensions
    if positions.ndim in {0, 1}:
        if screen_px.ndim != 0 and screen_px.shape != (1, ):
            raise ValueError('positions is 1-dimensional, but screen_px is not')
        if screen_cm.ndim != 0 and screen_cm.shape != (1, ):
            raise ValueError('positions is 1-dimensional, but screen_cm is not')
    if positions.ndim != 0 and positions.shape[-1] == 2:
        if screen_px.shape != (2, ):
            raise ValueError('positions is 2-dimensional, but screen_px is not')
        if screen_cm.shape != (2, ):
            raise ValueError('positions is 2-dimensional, but screen_cm is not')

    # compute eye-to-screen-distance in pixels
    distance_px = distance_cm * (screen_px / screen_cm)

    # center screen coordinates such that 0 is in the center of the screen
    if center_origin:
        positions = positions - (screen_px - 1) / 2

    # 180 / pi transforms arc measure to degrees
    return np.arctan2(positions, distance_px) * 180 / np.pi


def pos2vel(
        positions: list[float] | list[list[float]] | np.ndarray,
        sampling_rate: float = 1000,
        method: str = 'smooth',
) -> np.ndarray:
    """Compute velocity time series from 2-dimensional position time series.

    Adapted from Engbert et al.: Microsaccade Toolbox 0.9.

    Parameters
    ----------
    positions : array_like
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

    Returns
    -------
    velocities : array_like
        Velocity time series in input_unit / sec

    Raises
    ------
    ValueError
        If selected method is invalid, positions-array is too short for the
        selected method or the sampling rate is below zero

    """
    if sampling_rate <= 0:
        raise ValueError('sampling_rate needs to be above zero')

    pos = np.array(positions)

    if method == 'smooth' and pos.shape[0] < 6:
        raise ValueError(
            'positions has to have at least 6 elements for method "smooth"')
    if method == 'neighbors' and pos.shape[0] < 3:
        raise ValueError(
            'positions has to have at least 3 elements for method "neighbors"')
    if method == 'preceding' and pos.shape[0] < 2:
        raise ValueError(
            'positions has to have at least 2 elements for method "preceding"')

    N = pos.shape[0]
    vel = np.zeros(pos.shape)

    if method == 'smooth':
        moving_pos_avg = pos[4:N] + pos[3:N-1] - pos[1:N-3] - pos[0:N-4]
        # mean(pos_-2, pos_-1) and mean(pos_1, pos_2)
        # needs division by two
        # window is now 3 samples long (pos_-1.5, pos_0, pos_1+5)
        # we therefore need a divison by three, all in all it's a division by 6
        vel[2:N-2] = moving_pos_avg * sampling_rate / 6
        # for second and second last sample:
        # calculate velocity from preceding and subsequent sample
        vel[1] = (pos[2] - pos[0]) * sampling_rate / 2
        vel[N-2] = (pos[N-1] - pos[N-3]) * sampling_rate / 2

    elif method == 'neighbors':
        # window size is two, so we need to divide by two
        vel[1:N-1] = (pos[2:N] - pos[0:N-2]) * sampling_rate / 2

    elif method == 'preceding':
        vel[1:N] = (pos[1:N] - pos[0:N-1]) * sampling_rate

    else:
        valid_methods = ['smooth', 'neighbors', 'preceding']
        raise ValueError(f'Method needs to be in {valid_methods}'
                         f' (is: {method})')

    return vel
