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
This module holds basic checks which will be reused in other modules.
"""
from typing import Any

import numpy as np


def check_no_zeros(variable: Any, name: str = 'variable'):
    """
    Check if variable, or if it is iterable, any of its components are zero.
    """
    # construct error message first
    error_message = f'{name} must not be zero'

    # ducktyping check if variable is iterable
    try:
        _ = iter(variable)

    # variable is not iterable, simple check for zero
    except TypeError as exception:
        if variable == 0:
            raise ValueError(error_message) from exception

    # variable is iterable, check each element for zero
    else:
        error_message = 'each component in ' + error_message

        for variable_component in variable:
            if variable_component == 0:
                raise ValueError(error_message)


def check_nan_both_channels(arr: np.ndarray):
    """
    Checks if all nans occur at the same time steps for both channels.
    """
    # sanity check: horizontal and vertical gaze coordinates missing
    # values at the same time (Eyelink eyetracker never records only
    # one coordinate)
    if not np.array_equal(np.isnan(arr[:, 0]), np.isnan(arr[:, 1])):
        raise ValueError(
            'nans have to occur at the same steps of horizontal and vertical direction',
        )
