"""
This module tests functionality of the IDT algorithm

"""

import pytest
from pymovements.events.idt import idt


@pytest.mark.parametrize(
    'kwargs, expected_error',
    [
        pytest.param(
            {
                'positions': None,
                'dispersion_threshold': 1,
                'duration_threshold': 1,
            },
            ValueError,
            id='positions_none_raises_value_error'
        ),
        pytest.param(
            {
                'positions': [[1, 2], [1, 2]],
                'dispersion_threshold': None,
                'duration_threshold': 1,
            },
            TypeError,
            id='dispersion_threshold_none_raises_type_error'
        ),
        pytest.param(
            {
                'positions': [[1, 2], [1, 2]],
                'dispersion_threshold': 1,
                'duration_threshold': None,
            },
            TypeError,
            id='duration_threshold_none_raises_type_error'
        ),
        pytest.param(
            {
                'positions': 1,
                'dispersion_threshold': 1,
                'duration_threshold': 1,
            },
            ValueError,
            id='positions_not_array_like_raises_value_error'
        ),
        pytest.param(
            {
                'positions': [1, 2, 3],
                'dispersion_threshold': 1,
                'duration_threshold': 1,
            },
            ValueError,
            id='positions_1d_raises_value_error'
        ),
        pytest.param(
            {
                'positions': [[1, 2, 3], [1, 2, 3]],
                'dispersion_threshold': 1,
                'duration_threshold': 1,
            },
            ValueError,
            id='positions_not_2_elements_in_second_dimension_raises_value_error'
        ),
        pytest.param(
            {
                'positions': [[1, 2], [1, 2]],
                'dispersion_threshold': 0,
                'duration_threshold': 1,
            },
            ValueError,
            id='dispersion_threshold_not_greater_than_0_raises_value_error'
        ),
        pytest.param(
            {
                'positions': [[1, 2], [1, 2]],
                'dispersion_threshold': 1,
                'duration_threshold': 0,
            },
            ValueError,
            id='duration_threshold_not_greater_than_0_raises_value_error'
        ),
        pytest.param(
            {
                'positions': [[1, 2], [1, 2]],
                'dispersion_threshold': 1,
                'duration_threshold': 1.0,
            },
            TypeError,
            id='duration_threshold_not_integer_raises_type_error'
        )
    ]
)
def test_idt_raises_error(kwargs, expected_error):
    """Test if idt raises expected error."""
    with pytest.raises(expected_error):
        idt(**kwargs)
