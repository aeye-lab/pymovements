# Copyright (c) 2022-2023 The pymovements Project Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Test pymovements checks.
"""
import numpy as np
import pytest

from pymovements.utils.checks import check_nan_both_channels
from pymovements.utils.checks import check_no_zeros
from pymovements.utils.checks import check_shapes_positions_velocities


@pytest.mark.parametrize(
    'variable, expected_error',
    [
        pytest.param(5, None, id='non_zero_single_variable_raises_no_error'),
        pytest.param(0, ValueError, id='zero_single_variable_raises_value_error'),
        pytest.param([1, 2, 3], None, id='non_zero_list_raises_no_error'),
        pytest.param([1, 0, 3], ValueError, id='zero_list_raises_value_error'),
        pytest.param(np.array([1, 2, 3]), None, id='non_zero_np_array_raises_no_error'),
        pytest.param(np.array([1, 0, 3]), ValueError, id='zero_np_array_raises_value_error'),
    ],
)
def test_check_no_zeros_raises_error(variable, expected_error):
    """
    Test that check_no_zeros() only raises an Exception if there are zeros in the input array.
    """
    if expected_error is None:
        check_no_zeros(variable)
    else:
        with pytest.raises(expected_error):
            check_no_zeros(variable)


@pytest.mark.parametrize(
    'arr, expected_error',
    [
        pytest.param(
            np.array([[1, 2], [3, 4]]),
            None,
            id='no_nans_raises_no_error',
        ),
        pytest.param(
            np.array([[1, 2], [np.nan, np.nan]]),
            None,
            id='nans_same_time_steps_raises_no_error',
        ),
        pytest.param(
            np.array([[np.nan, 2], [np.nan, 4]]),
            ValueError,
            id='nans_different_time_steps_raises_value_error',
        ),
        pytest.param(
            np.array([[np.nan, 2], [3, 4]]),
            ValueError,
            id='nans_only_left_channel_raises_value_error',
        ),
        pytest.param(
            np.array([[1, np.nan], [3, 4]]),
            ValueError,
            id='nans_only_right_channel_raises_value_error',
        ),
    ],
)
def test_check_nan_both_channels_raises_error(arr, expected_error):
    """
    Test that check_nan_both_channels() only raises an Exception if all nans
    occur at the same time step for both channels.
    """
    if expected_error is None:
        check_nan_both_channels(arr)
    else:
        with pytest.raises(expected_error):
            check_nan_both_channels(arr)


# Test check_shapes_positions_velocities
@pytest.mark.parametrize(
    'kwargs, expected_error',
    [
        pytest.param(
            {
                'positions': np.array([[1, 2], [3, 4]]),
                'velocities': np.array([[1, 2], [3, 4]]),
            },
            None,
            id='positions_and_velocities_shape_N_2_raises_no_error',
        ),
        pytest.param(
            {
                'positions': np.array([[1, 2], [3, 4]]),
                'velocities': np.array([1, 2, 3, 4]),
            },
            ValueError,
            id='positions_shape_N_2_velocities_not_shape_N_2_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.array([1, 2, 3, 4]),
                'velocities': np.array([[1, 2], [3, 4]]),
            },
            ValueError,
            id='positions_not_shape_N_2_velocities_shape_N_2_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.array([1, 2, 3, 4]),
                'velocities': np.array([1, 2, 3, 4]),
            },
            ValueError,
            id='positions_and_velocities_not_shape_N_2_raises_value_error',
        ),
        pytest.param(
            {
                'positions': np.array([[1, 2], [3, 4]]),
                'velocities': np.array([[1, 2], [3, 4], [5, 6]]),
            },
            ValueError,
            id='positions_and_velocities_N_2_but_different_lengths_raises_value_error',
        ),
    ],
)
def test_check_shapes_positions_velocities_raises_error(kwargs, expected_error):
    """
    Test that check_shapes_positions_velocities() only raises an Exception if
    the shapes of the positions and velocities are not (N, 2) or if the lengths
    of the positions and velocities are not equal.
    """
    if expected_error is None:
        check_shapes_positions_velocities(**kwargs)
    else:
        with pytest.raises(expected_error):
            check_shapes_positions_velocities(**kwargs)
