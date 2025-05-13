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
            Experiment(sampling_rate=1000),
            Experiment(eyetracker=EyeTracker(sampling_rate=1000)),
            id='explicit_sampling_rate_and_eyetracker',
        ),
        pytest.param(
            Experiment(1024, 768),
            Experiment(screen=Screen(1024, 768)),
            id='explicit_screen_size_and_screen',
        ),
    ],
)
def test_sampling_rate_equality(experiment1, experiment2):
    assert experiment1 == experiment2


@pytest.mark.parametrize(
    ('dictionary', 'expected_experiment'),
    [
        pytest.param(
            {'sampling_rate': 1000},
            Experiment(eyetracker=EyeTracker(sampling_rate=1000)),
            id='sampling_rate',
        ),

        pytest.param(
            {'eyetracker': {'sampling_rate': 1000}},
            Experiment(eyetracker=EyeTracker(sampling_rate=1000)),
            id='eyetracker_sampling_rate',
        ),

        pytest.param(
            {'screen_width_px': 1024, 'screen_height_px': 768},
            Experiment(screen=Screen(1024, 768)),
            id='screen_width_px_and_screen_height_px',
        ),

        pytest.param(
            {'screen': {'width_px': 1024, 'height_px': 768}},
            Experiment(screen=Screen(1024, 768)),
            id='screen_width_px_and_height_px',
        ),
    ],
)
def test_experiment_from_dict(dictionary, expected_experiment):
    experiment = Experiment.from_dict(dictionary)
    assert experiment == expected_experiment


def test_experiment_to_dict_exclude_none():
    experiment = Experiment(
        origin=None,
        screen=Screen(),
        eyetracker=EyeTracker(),
    )
    dict_default = experiment.to_dict()
    assert 'screen' in dict_default
    assert 'origin' not in dict_default
    assert 'eyetracker' not in dict_default

    experiment.screen = Screen(origin=None)
    bool_screen_false = experiment.to_dict()
    assert 'screen' not in bool_screen_false

    experiment.eyetracker = EyeTracker(left=True)
    dict_non_default = experiment.to_dict(exclude_none=False)
    assert 'eyetracker' in dict_non_default
    assert 'screen' in dict_non_default


@pytest.mark.parametrize(
    ('experiment', 'expected_bool'),
    [
        pytest.param(
            Experiment(),
            False,
            marks=pytest.mark.xfail(reason='#1148'),
            id='default',
        ),

        pytest.param(
            Experiment(origin=None),
            False,
            id='origin_none',
        ),

        pytest.param(
            Experiment(origin='center'),
            True,
            id='origin_center',
        ),

        pytest.param(
            Experiment(distance=60),
            True,
            id='distance_60',
        ),
    ],
)
def test_experiment_bool(experiment, expected_bool):
    assert bool(experiment) == expected_bool
