"""
This module tests functionality of base event classes.
"""
from __future__ import annotations

import numpy as np
import pytest

from pymovements.events.ivt import ivt


@pytest.mark.parametrize(
    'kwargs, expected_error',
    [
        pytest.param(
            {
                'positions': None,
                'velocities': np.ones((100, 2)),
                'threshold': 1.
            },
            ValueError,
            id='none_positions_raises_value_error'
        ),
        pytest.param(
            {
                'positions': np.ones((100, 2)),
                'velocities': None,
                'threshold': 1.
            },
            ValueError,
            id='none_velocities_raises_value_error'
        ),
        pytest.param(
            {
                'positions': np.ones((100, 2)),
                'velocities': np.ones((100, 2)),
                'threshold': None
            },
            ValueError,
            id='none_threshold_raises_value_error'
        ),
    ]
)
def test_ivt_raise_error(kwargs, expected_error):
    with pytest.raises(expected_error):
        ivt(**kwargs)


@pytest.mark.parametrize(
    'kwargs, expected_count',
    [
        pytest.param(
            {
                'positions': np.ones((100, 2)),
                'velocities': np.ones((100, 2)),
                'threshold': 1.5
            },
            1,
            id='constant_velocity_below_threshold_returns_single_fixation'
        ),
        pytest.param(
            {
                'positions': np.ones((100, 2)),
                'velocities': np.ones((100, 2)),
                'threshold': 1.
            },
            0,
            id='constant_velocity_above_threshold_returns_no_fixation'
        ),

    ]
)
def test_ivt_returns_count(kwargs, expected_count):
    assert len(ivt(**kwargs)) == expected_count
