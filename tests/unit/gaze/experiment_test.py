# Copyright (c) 2024-2025 The pymovements Project Authors
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

from pymovements import Experiment
from pymovements import EyeTracker
from pymovements import Screen


def test_sampling_rate_setter():
    experiment = Experiment(1280, 1024, 38, 30, sampling_rate=1000.0)
    assert experiment.sampling_rate == 1000.0

    experiment.sampling_rate = 100.0
    assert experiment.sampling_rate == 100.0


@pytest.mark.parametrize(
    'experiment_init_kwargs',
    [
        pytest.param(
            {},
            id='empty',
        ),
        pytest.param(
            {'sampling_rate': 1000},
            id='only_sampling_rate',
        ),
        pytest.param(
            {'screen': Screen(1024, 768)},
            id='only_screen',
        ),
        pytest.param(
            {'screen': Screen(1024, 768), 'eyetracker': EyeTracker(sampling_rate=1000)},
            id='screen_and_eyetracker',
        ),

    ],
)
def test_sampling_rate_trivial_equality(experiment_init_kwargs):
    experiment1 = Experiment(**experiment_init_kwargs)
    experiment2 = Experiment(**experiment_init_kwargs)
    assert experiment1 == experiment2


@pytest.mark.parametrize(
    ('experiment1', 'experiment2'),
    [
        pytest.param(
            pm.Experiment(sampling_rate=1000),
            pm.Experiment(eyetracker=EyeTracker(sampling_rate=1000)),
            id='explicit_sampling_rate_and_eyetracker',
        ),
        pytest.param(
            pm.Experiment(1024, 768),
            pm.Experiment(screen=Screen(1024, 768)),
            id='explicit_screen_size_and_screen',
        ),
    ],
)
def test_sampling_rate_equality(experiment1, experiment2):
    assert experiment1 == experiment2
