# Copyright (c) 2023 The pymovements Project Authors
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
"""Test for EyeTracker class."""
import pytest

from pymovements.gaze.eyetracker import EyeTracker


def test_eyetracker_with_sampling_rate():
    EyeTracker(
        1000.0, False, True, 'EyeLink 1000 Plus',
        '1.5.3', 'Arm Mount / Monocular / Remote',
    )


@pytest.mark.parametrize(
    ('sampling_rate', 'exception', 'exception_msg'),
    [
        pytest.param(
            None,
            TypeError,
            "'sampling_rate' must not be None",
            id='no_sampling_rate',
        ),
        pytest.param(
            -1000.0,
            ValueError,
            "'sampling_rate' must be greater than zero but is -1000.0",
            id='negative_sampling_rate',
        ),
    ],
)
def test_eyetracker_with_invalid_sampling_rate(sampling_rate, exception, exception_msg):
    with pytest.raises(exception) as exc_info:
        EyeTracker(sampling_rate)

    msg, = exc_info.value.args
    assert msg == exception_msg
