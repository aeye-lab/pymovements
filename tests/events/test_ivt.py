# Copyright (c) 2022 The pymovements Project Authors
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
