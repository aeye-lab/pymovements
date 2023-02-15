"""This module tests functionality of the IDT algorithm."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.events.idt import idt
from pymovements.synthetic import step_function


@pytest.mark.parametrize(
    'kwargs, expected_error',
    [
        pytest.param(
            {
                'positions': None,
                'dispersion_threshold': 1,
                'minimum_duration': 1,
            },
            ValueError,
            id='positions_none_raises_value_error',
        ),
        pytest.param(
            {
                'positions': [[1, 2], [1, 2]],
                'dispersion_threshold': None,
                'minimum_duration': 1,
            },
            TypeError,
            id='dispersion_threshold_none_raises_type_error',
        ),
        pytest.param(
            {
                'positions': [[1, 2], [1, 2]],
                'dispersion_threshold': 1,
                'minimum_duration': None,
            },
            TypeError,
            id='duration_threshold_none_raises_type_error',
        ),
        pytest.param(
            {
                'positions': 1,
                'dispersion_threshold': 1,
                'minimum_duration': 1,
            },
            ValueError,
            id='positions_not_array_like_raises_value_error',
        ),
        pytest.param(
            {
                'positions': [1, 2, 3],
                'dispersion_threshold': 1,
                'minimum_duration': 1,
            },
            ValueError,
            id='positions_1d_raises_value_error',
        ),
        pytest.param(
            {
                'positions': [[1, 2, 3], [1, 2, 3]],
                'dispersion_threshold': 1,
                'minimum_duration': 1,
            },
            ValueError,
            id='positions_not_2_elements_in_second_dimension_raises_value_error',
        ),
        pytest.param(
            {
                'positions': [[1, 2], [1, 2]],
                'dispersion_threshold': 0,
                'minimum_duration': 1,
            },
            ValueError,
            id='dispersion_threshold_not_greater_than_0_raises_value_error',
        ),
        pytest.param(
            {
                'positions': [[1, 2], [1, 2]],
                'dispersion_threshold': 1,
                'minimum_duration': 0,
            },
            ValueError,
            id='duration_threshold_not_greater_than_0_raises_value_error',
        ),
        pytest.param(
            {
                'positions': [[1, 2], [1, 2]],
                'dispersion_threshold': 1,
                'minimum_duration': 1.0,
            },
            TypeError,
            id='duration_threshold_not_integer_raises_type_error',
        ),
    ],
)
def test_idt_raises_error(kwargs, expected_error):
    """Test if idt raises expected error."""
    with pytest.raises(expected_error):
        idt(**kwargs)


@pytest.mark.parametrize(
    'kwargs, expected',
    [
        pytest.param(
            {
                'positions': step_function(length=100, steps=[0], values=[(0, 0)]),
                'dispersion_threshold': 1,
                'minimum_duration': 1,
            },
            pl.DataFrame(
                {
                    'type': ['fixation'],
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
                'dispersion_threshold': 1,
                'minimum_duration': 1,
            },
            pl.DataFrame(
                {
                    'type': 'fixation',
                    'onset': [0, 50],
                    'offset': [50, 99],  # should be [49, 99]
                    'position': [(0.18, 0.18), (1.0, 1.0)],  # should be: [(0.0, 0.0), (1.0, 1.0)]
                },
            ),
            id='three_steps_two_fixations',
        ),
    ],
)
def test_idt_detects_fixations(kwargs, expected):
    events = idt(**kwargs)

    assert_frame_equal(events, expected)
