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
"""Test for Experiment class."""
import pytest

from pymovements.gaze.experiment import Experiment
from pymovements.gaze.eyetracker import EyeTracker


def test_experiment_without_sampling_rate():
    with pytest.raises(TypeError):
        eyetracker = EyeTracker(
            None, False, True, 'EyeLink 1000 Plus', '1.5.3', 'Arm Mount / Monocular / Remote',
        )
        Experiment(
            1280, 1024, 38, 30, 68, 'lower left', None, eyetracker=eyetracker,
        )


def test_experiment_with_sampling_rate():
    eyetracker = EyeTracker(
        1000.0, False, True, 'EyeLink 1000 Plus', '1.5.3', 'Arm Mount / Monocular / Remote',
    )
    experiment = Experiment(
        1280, 1024, 38, 30, 68, 'lower left', eyetracker=eyetracker,
    )
    assert experiment.sampling_rate == 1000.0
