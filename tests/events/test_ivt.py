"""This module tests functionality of the IVT algorithm."""
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.events.ivt import ivt
from pymovements.synthetic import step_function
from pymovements.transforms import pos2vel


@pytest.mark.parametrize(
    'kwargs, expected_error',
    [
        pytest.param(
            {
                'positions': None,
                'velocities': np.ones((100, 2)),
                'velocity_threshold': 1.,
                'minimum_duration': 1,
            },
            ValueError,
            id='positions_none_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.ones((100, 2)),
                'velocities': None,
                'velocity_threshold': 1.,
                'minimum_duration': 1,
            },
            ValueError,
            id='velocities_none_raises_value_error',
        ),
        pytest.param(
            {
                'positions': 1,
                'velocities': np.ones((100, 2)),
                'velocity_threshold': 1.,
                'minimum_duration': 1,
            },
            ValueError,
            id='positions_not_array_like_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.ones((100, 2)),
                'velocities': 1,
                'velocity_threshold': 1.,
                'minimum_duration': 1,
            },
            ValueError,
            id='velocities_not_array_like_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.ones(100),
                'velocities': np.ones((100, 2)),
                'velocity_threshold': 1.,
                'minimum_duration': 1,
            },
            ValueError,
            id='positions_not_2d_array_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.ones((100, 2)),
                'velocities': np.ones(100),
                'velocity_threshold': 1.,
                'minimum_duration': 1,
            },
            ValueError,
            id='velocities_not_2d_array_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.ones((100, 3)),
                'velocities': np.ones((100, 2)),
                'velocity_threshold': 1.,
                'minimum_duration': 1,
            },
            ValueError,
            id='positions_not_2_elements_in_second_dimension_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.ones((100, 2)),
                'velocities': np.ones((100, 3)),
                'velocity_threshold': 1.,
                'minimum_duration': 1,
            },
            ValueError,
            id='velocities_not_2_elements_in_second_dimension_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.ones((100, 2)),
                'velocities': np.ones((101, 2)),
                'velocity_threshold': 1.,
                'minimum_duration': 1,
            },
            ValueError,
            id='positions_and_velocities_different_lengths_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.ones((100, 2)),
                'velocities': np.ones((100, 2)),
                'velocity_threshold': None,
                'minimum_duration': 1,
            },
            ValueError,
            id='velocity_threshold_none_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.ones((100, 2)),
                'velocities': np.ones((100, 2)),
                'velocity_threshold': '1.',
                'minimum_duration': 1,
            },
            TypeError,
            id='velocity_threshold_not_float_raises_type_error',
        ),
        pytest.param(
            {
                'positions': np.ones((100, 2)),
                'velocities': np.ones((100, 2)),
                'velocity_threshold': 0.,
                'minimum_duration': 1,
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


@pytest.mark.parametrize(
    'kwargs, expected',
    [
        pytest.param(
            {
                'positions': step_function(length=100, steps=[0], values=[(0, 0)]),
                'velocity_threshold': 1,
                'minimum_duration': 1,
            },
            pl.DataFrame(
                {
                    'type': 'fixation',
                    'onset': [0],
                    'offset': [99],
                    'position': [(0.0, 0.0)],
                },
            ),
            id='constant_position_single_fixation',
        ),
        pytest.param(
            {
                'positions': step_function(
                    length=100,
                    steps=[49, 50],
                    values=[(9, 9), (1, 1)],
                    start_value=(0, 0),
                ),
                'velocity_threshold': 1,
                'minimum_duration': 1,
            },
            pl.DataFrame(
                {
                    'type': 'fixation',
                    'onset': [0, 51],
                    'offset': [48, 99],
                    'position': [(0.0, 0.0), (1.0, 1.0)],
                },
            ),
            id='three_steps_two_fixations',
        ),
    ],
)
def test_idt_detects_fixations(kwargs, expected):
    velocities = pos2vel(kwargs['positions'], sampling_rate=10, method='preceding')
    events = ivt(velocities=velocities, **kwargs)

    assert_frame_equal(events, expected)
