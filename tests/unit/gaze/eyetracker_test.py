# Copyright (c) 2023-2025 The pymovements Project Authors
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


def test_eyetracker_with_positive_sampling_rate():
    EyeTracker(
        1000.0, False, True, 'EyeLink 1000 Plus',
        '1.5.3', 'Arm Mount / Monocular / Remote',
    )


def test_eyetracker_with_negative_sampling_rate():
    with pytest.raises(ValueError):
        EyeTracker(
            -500.0, False, True, 'EyeLink 1000 Plus',
            '1.5.3', 'Arm Mount / Monocular / Remote',
        )


def test_eyetracker_without_sampling_rate():
    EyeTracker(
        None, False, True, 'EyeLink 1000 Plus',
        '1.5.3', 'Arm Mount / Monocular / Remote',
    )


def test_eyetracker_to_dict_exclude_none():
    eyetracker = EyeTracker(None, False,)
    dict_default = eyetracker.to_dict()
    assert 'sampling_rate' not in dict_default
    assert 'left' not in dict_default

    dict_non_default = eyetracker.to_dict(exclude_none=False)
    assert 'sampling_rate' in dict_non_default
    assert 'left' in dict_non_default


def test_eyetracker_bool_all_none():
    assert not bool(EyeTracker())
