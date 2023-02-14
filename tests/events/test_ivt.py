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
                'velocity_threshold': 1.,
            },
            ValueError,
            id='positions_none_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.ones((100, 2)),
                'velocities': None,
                'velocity_threshold': 1.,
            },
            ValueError,
            id='velocities_none_raises_value_error',
        ),
        pytest.param(
            {
                'positions': 1,
                'velocities': np.ones((100, 2)),
                'velocity_threshold': 1.,
            },
            ValueError,
            id='positions_not_array_like_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.ones((100, 2)),
                'velocities': 1,
                'velocity_threshold': 1.,
            },
            ValueError,
            id='velocities_not_array_like_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.ones(100),
                'velocities': np.ones((100, 2)),
                'velocity_threshold': 1.,
            },
            ValueError,
            id='positions_not_2d_array_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.ones((100, 2)),
                'velocities': np.ones(100),
                'velocity_threshold': 1.,
            },
            ValueError,
            id='velocities_not_2d_array_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.ones((100, 3)),
                'velocities': np.ones((100, 2)),
                'velocity_threshold': 1.,
            },
            ValueError,
            id='positions_not_2_elements_in_second_dimension_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.ones((100, 2)),
                'velocities': np.ones((100, 3)),
                'velocity_threshold': 1.,
            },
            ValueError,
            id='velocities_not_2_elements_in_second_dimension_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.ones((100, 2)),
                'velocities': np.ones((101, 2)),
                'velocity_threshold': 1.,
            },
            ValueError,
            id='positions_and_velocities_different_lengths_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.ones((100, 2)),
                'velocities': np.ones((100, 2)),
                'velocity_threshold': None,
            },
            ValueError,
            id='velocity_threshold_none_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.ones((100, 2)),
                'velocities': np.ones((100, 2)),
                'velocity_threshold': '1.',
            },
            TypeError,
            id='velocity_threshold_not_float_raises_type_error',
        ),
        pytest.param(
            {
                'positions': np.ones((100, 2)),
                'velocities': np.ones((100, 2)),
                'velocity_threshold': 0.,
            },
            ValueError,
            id='velocity_threshold_not_greater_than_0_raises_value_error',
        ),
    ],
)
def test_ivt_raise_error(kwargs, expected_error):
    """Test if ivt raises expected error."""
    with pytest.raises(expected_error):
        ivt(**kwargs)
