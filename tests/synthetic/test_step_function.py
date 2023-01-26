"""
This module tests functionality of the synthetic eye gaze step function.
"""
import numpy as np
import pytest
from pymovements.synthetic import step_function


@pytest.mark.parametrize(
    'params, expected',
    [
        pytest.param(
            {'length': 0, 'steps': [0], 'values': [0]},
            {'value': np.array([])},
            id='length_0_returns_empty_array',
        ),
        pytest.param(
            {'length': 10, 'steps': [0], 'values': [1], 'start_value': 0},
            {'value': np.ones(10)},
            id='length_10_with_step_at_start',
        ),
        pytest.param(
            {'length': 10, 'steps': [5], 'values': [0], 'start_value': 1},
            {'value': np.concatenate([np.ones(5), np.zeros(5)])},
            id='length_10_start_value_1_step_5_to_0',
        ),
        pytest.param(
            {'length': 100, 'steps': [10, 50, 90], 'values': [1, 0, 20], 'start_value': 0},
            {'value': np.concatenate([np.zeros(10), np.ones(40), np.zeros(40), np.ones(10) * 20])},
            id='length_100_3_steps',
        ),
        pytest.param(
            {'length': 10, 'steps': [5], 'values': [(1, 2)], 'start_value': 10},
            {'value': np.concatenate([np.tile(10, (5, 2)), np.tile((1, 2), (5, 1))])},
            id='length_10_2_channel_single_step_with_single_start_value',
        ),
        pytest.param(
            {'length': 10, 'steps': [5], 'values': [(1, 2)], 'start_value': (11, 12)},
            {'value': np.concatenate([np.tile((11, 12), (5, 1)), np.tile((1, 2), (5, 1))])},
            id='length_10_2_channel_single_step_with_2_channel_start_value',
        ),
        pytest.param(
            {'length': 10, 'steps': [5], 'values': [(1, 2, 3, 4)], 'start_value': (11, 12, 13, 14)},
            {
                'value': np.concatenate([
                    np.tile((11, 12, 13, 14), (5, 1)),
                    np.tile((1, 2, 3, 4), (5, 1)),
                ]),
            },
            id='length_10_4_channel_single_step_with_start_value',
        ),
        pytest.param(
            {'length': 100, 'steps': [10, 50, 90], 'values': [1, 0], 'start_value': 0},
            {'exception': ValueError},
            id='steps_values_unequal_length_raises_value_error',
        ),
        pytest.param(
            {'length': 100, 'steps': [10, 90, 50], 'values': [1, 0, 1], 'start_value': 0},
            {'exception': ValueError},
            id='steps_not_sorted_raises_value_error',
        ),
        pytest.param(
            {'length': 10, 'steps': [3, 5], 'values': [(1, 2), (3, 5, 6)]},
            {'exception': ValueError},
            id='varying_number_of_channels_raises_value_error',
        ),
        pytest.param(
            {'length': 10, 'steps': [5], 'values': [(1, 2)], 'start_value': (1, 2, 3)},
            {'exception': ValueError},
            id='number_of_channels_unequal_start_value_channels_raises_value_error',
        ),
    ],
)
def test_step_function(params, expected):
    """Test step function."""

    if 'exception' in expected:
        with pytest.raises(expected['exception']):
            step_function(**params)
        return

    arr = step_function(**params)
    assert np.array_equal(arr, expected['value']), f"arr = {arr}, expected = {expected['value']}"
