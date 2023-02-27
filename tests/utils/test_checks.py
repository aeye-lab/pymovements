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
import pytest
import numpy as np

from pymovements.utils.checks import check_no_zeros
from pymovements.utils.checks import check_nan_both_channels


@pytest.mark.parametrize(
    'variable, expected_error',
    [
        pytest.param(5, None, id='non_zero_single_variable'),
        pytest.param(0, ValueError, id='zero_single_variable'),
        pytest.param([1, 2, 3], None, id='non_zero_list'),
        pytest.param([1, 0, 3], ValueError, id='zero_list'),
        pytest.param(np.array([1, 2, 3]), None, id='non_zero_np_array'),
        pytest.param(np.array([1, 0, 3]), ValueError, id='zero_np_array')
    ]
)
def test_check_no_zeros_exception(variable, expected_error):
    """
    Test that check_no_zeros() only raises an Exception iff there are zeros in the input array.
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
            np.array([[1, 2], [3, 4]]), None, id='no_nans'
        ),
        pytest.param(
            np.array([[1, 2], [np.nan, np.nan]]), None, id='nans_same_time_steps'
        ),
        pytest.param(
            np.array([[np.nan, 2], [np.nan, 4]]), ValueError, id='nans_different_time_steps'
        ),
        pytest.param(
            np.array([[np.nan, 2], [3, 4]]), ValueError, id='nans_only_left_channel'
        ),
        pytest.param(
            np.array([[1, np.nan], [3, 4]]), ValueError, id='nans_only_right_channel'
        )
    ]
)
def test_check_nan_both_channels(arr, expected_error):
    """
    Test that check_nan_both_channels() only raises an Exception if all nans
    occur at the same time step for both channels.
    """
    if expected_error is None:
        check_nan_both_channels(arr)
    else:
        with pytest.raises(expected_error):
            check_nan_both_channels(arr)
