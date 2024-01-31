# Copyright (c) 2023-2024 The pymovements Project Authors
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
"""Test GazeDataFrame transform method."""
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.fixture(name='experiment')
def fixture_experiment():
    return pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000)


@pytest.mark.parametrize(
    ('gaze_init_kwargs', 'transform_method', 'transform_kwargs', 'expected'),
    [
        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'time': np.arange(1000, 1100, 1),
                        'x_pix': np.arange(0, 10, 0.1),
                        'y_pix': np.arange(20, 30, 0.1),
                    },
                ),
                'pixel_columns': ['x_pix', 'y_pix'],
            },
            pm.gaze.transforms.downsample, {'factor': 1},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {
                        'time': np.arange(1000, 1100, 1),
                        'x_pix': np.arange(0, 10, 0.1),
                        'y_pix': np.arange(20, 30, 0.1),
                    },
                ),
                pixel_columns=['x_pix', 'y_pix'],
            ),
            id='downsample_factor_1_method_pass',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'time': np.arange(1000, 1100, 1),
                        'x_pix': np.arange(0, 10, 0.1),
                        'y_pix': np.arange(20, 30, 0.1),
                    },
                ),
                'pixel_columns': ['x_pix', 'y_pix'],
            },
            'downsample', {'factor': 1},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {
                        'time': np.arange(1000, 1100, 1),
                        'x_pix': np.arange(0, 10, 0.1),
                        'y_pix': np.arange(20, 30, 0.1),
                    },
                ),
                pixel_columns=['x_pix', 'y_pix'],
            ),
            id='downsample_factor_1',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'time': np.arange(1000, 1010, 1),
                        'x_pix': np.arange(0, 1, 0.1),
                        'y_pix': np.arange(20, 21, 0.1),
                    },
                ),
                'pixel_columns': ['x_pix', 'y_pix'],
            },
            'downsample', {'factor': 2},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {
                        'time': np.arange(1000, 1010, 2),
                        'x_pix': np.arange(0, 1, 0.2),
                        'y_pix': [20.0, 20.2, 20.4, 20.6, 20.8],
                    },
                ),
                pixel_columns=['x_pix', 'y_pix'],
            ),
            id='downsample_factor_2',
        ),
        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'time': [1000],
                        'x_pix': [(100 - 1) / 2],
                        'y_pix': [0.0],
                    },
                ),
                'experiment': pm.Experiment(
                    sampling_rate=1000,
                    screen_width_px=100,
                    screen_height_px=100,
                    screen_width_cm=100,
                    screen_height_cm=100,
                    distance_cm=100,
                    origin='center',
                ),
                'pixel_columns': ['x_pix', 'y_pix'],
            },
            'center_origin', {},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {
                        'time': [1000],
                        'x_pix': [(100 - 1) / 2],
                        'y_pix': [0.0],
                    },
                ),
                pixel_columns=['x_pix', 'y_pix'],
            ),
            id='center_origin_center',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'time': [1000, 1000],
                        'x_pix': [(100 - 1) / 2, (100 - 1) / 2],
                        'y_pix': [0.0, 0.0],
                    },
                ),
                'experiment': pm.Experiment(
                    sampling_rate=1000,
                    screen_width_px=100,
                    screen_height_px=100,
                    screen_width_cm=100,
                    screen_height_cm=100,
                    distance_cm=100,
                    origin='lower left',
                ),
                'pixel_columns': ['x_pix', 'y_pix'],
            },
            'center_origin', {},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {
                        'time': [1000, 1000],
                        'x_pix': [0.0, 0.0],
                        'y_pix': [-(100 - 1) / 2, -(100 - 1) / 2],
                    },
                ),
                pixel_columns=['x_pix', 'y_pix'],
            ),
            id='center_origin_lower_left',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'time': [1000, 1000],
                        'x_pix': [(100 - 1) / 2, (100 - 1) / 2],
                        'y_pix': [0.0, 0.0],
                    },
                ),
                'experiment': pm.Experiment(
                    sampling_rate=1000,
                    screen_width_px=100,
                    screen_height_px=100,
                    screen_width_cm=100,
                    screen_height_cm=100,
                    distance_cm=100,
                    origin='center',
                ),
                'pixel_columns': ['x_pix', 'y_pix'],
            },
            'pix2deg', {},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {
                        'time': [1000, 1000],
                        'x_pix': [49.5, 49.5],
                        'y_pix': [0.0, 0.0],
                        'x_dva': [26.3354, 26.3354],
                        'y_dva': [0.0, 0.0],
                    },
                ),
                pixel_columns=['x_pix', 'y_pix'],
                position_columns=['x_dva', 'y_dva'],
            ),
            id='pix2deg_origin_center',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'time': [1000, 1000],
                        'x_pix': [(100 - 1) / 2, (100 - 1) / 2],
                        'y_pix': [0.0, 0.0],
                    },
                ),
                'experiment': pm.Experiment(
                    sampling_rate=1000,
                    screen_width_px=100,
                    screen_height_px=100,
                    screen_width_cm=100,
                    screen_height_cm=100,
                    distance_cm=100,
                    origin='center',
                ),
                'pixel_columns': ['x_pix', 'y_pix'],
            },
            'pix2deg', {'n_components': 2},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {
                        'time': [1000, 1000],
                        'x_pix': [49.5, 49.5],
                        'y_pix': [0.0, 0.0],
                        'x_dva': [26.3354, 26.3354],
                        'y_dva': [0.0, 0.0],
                    },
                ),
                pixel_columns=['x_pix', 'y_pix'],
                position_columns=['x_dva', 'y_dva'],
            ),
            id='pix2deg_origin_center_explicit_n_components',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'time': [1000],
                        'x_pix': [(100 - 1) / 2],
                        'y_pix': [0.0],
                    },
                ),
                'experiment': pm.Experiment(
                    sampling_rate=1000,
                    screen_width_px=100,
                    screen_height_px=100,
                    screen_width_cm=100,
                    screen_height_cm=100,
                    distance_cm=100,
                    origin='lower left',
                ),
                'pixel_columns': ['x_pix', 'y_pix'],
            },
            'pix2deg', {},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {
                        'time': [1000],
                        'x_pix': [49.5],
                        'y_pix': [0.0],
                        'x_dva': [0.0],
                        'y_dva': [-26.3354],
                    },
                ),
                pixel_columns=['x_pix', 'y_pix'],
                position_columns=['x_dva', 'y_dva'],
            ),
            id='pix2deg_origin_lower_left',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'time': [1000, 1000],
                        'x_pix': [(100 - 1) / 2, (100 - 1) / 2],
                        'y_pix': [0.0, 0.0],
                        'distance': [1000, 1000],
                    },
                ),
                'experiment': pm.Experiment(
                    sampling_rate=1000,
                    screen_width_px=100,
                    screen_height_px=100,
                    screen_width_cm=100,
                    screen_height_cm=100,
                    distance_cm=None,
                    origin='center',
                ),
                'pixel_columns': ['x_pix', 'y_pix'],
                'distance_column': 'distance',
            },
            'pix2deg', {},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {
                        'time': [1000, 1000],
                        'x_pix': [49.5, 49.5],
                        'y_pix': [0.0, 0.0],
                        'x_dva': [26.3354, 26.3354],
                        'y_dva': [0.0, 0.0],
                        'distance': [1000, 1000],
                    },
                ),
                pixel_columns=['x_pix', 'y_pix'],
                position_columns=['x_dva', 'y_dva'],
            ),
            id='pix2deg_origin_center_distance_column',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'time': [1000, 1000],
                        'x_pix': [(100 - 1) / 2, (100 - 1) / 2],
                        'y_pix': [0.0, 0.0],
                        'distance': [1000, 1000],
                    },
                ),
                'experiment': pm.Experiment(
                    sampling_rate=1000,
                    screen_width_px=100,
                    screen_height_px=100,
                    screen_width_cm=100,
                    screen_height_cm=100,
                    distance_cm=None,
                    origin='center',
                ),
                'pixel_columns': ['x_pix', 'y_pix'],
                'distance_column': 'distance',
            },
            'pix2deg', {'n_components': 2},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {
                        'time': [1000, 1000],
                        'x_pix': [49.5, 49.5],
                        'y_pix': [0.0, 0.0],
                        'x_dva': [26.3354, 26.3354],
                        'y_dva': [0.0, 0.0],
                        'distance': [1000, 1000],
                    },
                ),
                pixel_columns=['x_pix', 'y_pix'],
                position_columns=['x_dva', 'y_dva'],
                distance_column='distance',
            ),
            id='pix2deg_origin_center_explicit_n_components_distance_column',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'time': [1000, 1000],
                        'x_pix': [(100 - 1) / 2, (100 - 1) / 2],
                        'y_pix': [0.0, 0.0],
                        'distance': [1000, 1000],
                    },
                ),
                'experiment': pm.Experiment(
                    sampling_rate=1000,
                    screen_width_px=100,
                    screen_height_px=100,
                    screen_width_cm=100,
                    screen_height_cm=100,
                    distance_cm=None,
                    origin='center',
                ),
                'pixel_columns': ['x_pix', 'y_pix'],
            },
            'pix2deg', {'distance': 'distance'},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {
                        'time': [1000, 1000],
                        'x_pix': [49.5, 49.5],
                        'y_pix': [0.0, 0.0],
                        'x_dva': [26.3354, 26.3354],
                        'y_dva': [0.0, 0.0],
                        'distance': [1000, 1000],
                    },
                ),
                pixel_columns=['x_pix', 'y_pix'],
                position_columns=['x_dva', 'y_dva'],
            ),
            id='pix2deg_origin_center_explicit_distance_column',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'time': [1000],
                        'x_pix': [(100 - 1) / 2],
                        'y_pix': [0.0],
                        'distance': [1000],
                    },
                ),
                'experiment': pm.Experiment(
                    sampling_rate=1000,
                    screen_width_px=100,
                    screen_height_px=100,
                    screen_width_cm=100,
                    screen_height_cm=100,
                    distance_cm=None,
                    origin='lower left',
                ),
                'pixel_columns': ['x_pix', 'y_pix'],
                'distance_column': 'distance',
            },
            'pix2deg', {},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {
                        'time': [1000],
                        'x_pix': [49.5],
                        'y_pix': [0.0],
                        'x_dva': [0.0],
                        'y_dva': [-26.3354],
                        'distance': [1000],
                    },
                ),
                pixel_columns=['x_pix', 'y_pix'],
                position_columns=['x_dva', 'y_dva'],
                distance_column='distance',
            ),
            id='pix2deg_origin_lower_left_distance_column',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'trial_id': [1, 1, 1, 2, 2, 2],
                        'time': [1000, 1001, 1002, 1003, 1004, 1005],
                        'x_dva': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        'y_dva': [1.0, 1.1, 1.2, 1.0, 1.1, 1.2],
                    },
                ),
                'experiment': pm.Experiment(
                    sampling_rate=1000,
                    screen_width_px=100,
                    screen_height_px=100,
                    screen_width_cm=100,
                    screen_height_cm=100,
                    distance_cm=100,
                    origin='lower left',
                ),
                'position_columns': ['x_dva', 'y_dva'],
            },
            'pos2vel', {'method': 'preceding'},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {
                        'trial_id': [1, 1, 1, 2, 2, 2],
                        'time': [1000, 1001, 1002, 1003, 1004, 1005],
                        'x_dva': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        'y_dva': [1.0, 1.1, 1.2, 1.0, 1.1, 1.2],
                        'x_vel': [None, 0.0, 0.0, 0.0, 0.0, 0.0],
                        'y_vel': [None, 100.0, 100.0, -200.0, 100.0, 100.0],
                    },
                ),
                position_columns=['x_dva', 'y_dva'],
                velocity_columns=['x_vel', 'y_vel'],
            ),
            id='pos2vel_preceding',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'time': [1000, 1001, 1002],
                        'x_dva': [1.0, 1.0, 1.0],
                        'y_dva': [1.0, 1.1, 1.2],
                    },
                ),
                'experiment': pm.Experiment(
                    sampling_rate=1000,
                    screen_width_px=100,
                    screen_height_px=100,
                    screen_width_cm=100,
                    screen_height_cm=100,
                    distance_cm=100,
                    origin='lower left',
                ),
                'position_columns': ['x_dva', 'y_dva'],
            },
            'pos2vel', {'method': 'neighbors'},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {
                        'time': [1000, 1001, 1002],
                        'x_dva': [1.0, 1.0, 1.0],
                        'y_dva': [1.0, 1.1, 1.2],
                        'x_vel': [None, 0.0, None],
                        'y_vel': [None, 100.0, None],
                    },
                ),
                position_columns=['x_dva', 'y_dva'],
                velocity_columns=['x_vel', 'y_vel'],
            ),
            id='pos2vel_neighbors',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'time': [1000, 1001, 1002, 1003, 1004],
                        'x_dva': [1.0, 1.0, 1.0, 1.0, 1.0],
                        'y_dva': [1.0, 1.1, 1.2, 1.3, 1.4],
                    },
                ),
                'experiment': pm.Experiment(
                    sampling_rate=1000,
                    screen_width_px=100,
                    screen_height_px=100,
                    screen_width_cm=100,
                    screen_height_cm=100,
                    distance_cm=100,
                    origin='lower left',
                ),
                'position_columns': ['x_dva', 'y_dva'],
            },
            'pos2vel', {'method': 'fivepoint'},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {
                        'time': [1000, 1001, 1002, 1003, 1004],
                        'x_dva': [1.0, 1.0, 1.0, 1.0, 1.0],
                        'y_dva': [1.0, 1.1, 1.2, 1.3, 1.4],
                        'x_vel': [None, None, 0.0, None, None],
                        'y_vel': [None, None, 100.0, None, None],
                    },
                ),
                position_columns=['x_dva', 'y_dva'],
                velocity_columns=['x_vel', 'y_vel'],
            ),
            id='pos2vel_five_point',
        ),

        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'trial_id': [1, 1, 1, 2, 2, 2],
                        'time': [1000, 1001, 1002, 1003, 1004, 1005],
                        'x_dva': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        'y_dva': [1.0, 1.1, 1.2, 1.0, 1.1, 1.2],
                    },
                ),
                'experiment': pm.Experiment(
                    sampling_rate=1000,
                    screen_width_px=100,
                    screen_height_px=100,
                    screen_width_cm=100,
                    screen_height_cm=100,
                    distance_cm=100,
                    origin='lower left',
                ),
                'position_columns': ['x_dva', 'y_dva'],
                'trial_columns': 'trial_id',
            },
            'pos2vel', {'method': 'preceding'},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {
                        'trial_id': [1, 1, 1, 2, 2, 2],
                        'time': [1000, 1001, 1002, 1003, 1004, 1005],
                        'x_dva': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        'y_dva': [1.0, 1.1, 1.2, 1.0, 1.1, 1.2],
                        'x_vel': [None, 0.0, 0.0, None, 0.0, 0.0],
                        'y_vel': [None, 100.0, 100.0, None, 100.0, 100.0],
                    },
                ),
                position_columns=['x_dva', 'y_dva'],
                velocity_columns=['x_vel', 'y_vel'],
            ),
            id='pos2vel_preceding_trialize_single_column_str',
        ),
        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'x_dva': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        'y_dva': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                    },
                ),
                'position_columns': ['x_dva', 'y_dva'],
            },
            'smooth', {'method': 'moving_average', 'window_length': 3},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {
                        'x_dva': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        'y_dva': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                    },
                ),
                position_columns=['x_dva', 'y_dva'],
            ),
            id='smooth',
        ),
        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'x_dva': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        'y_dva': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                    },
                ),
                'position_columns': ['x_dva', 'y_dva'],
            },
            pm.gaze.transforms.smooth, {'method': 'moving_average', 'window_length': 3},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {
                        'x_dva': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        'y_dva': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                    },
                ),
                position_columns=['x_dva', 'y_dva'],
            ),
            id='smooth_method_pass',
        ),

    ],
)
def test_gaze_transform_expected_frame(
        gaze_init_kwargs, transform_method, transform_kwargs, expected,
):
    gaze = pm.GazeDataFrame(**gaze_init_kwargs)
    gaze.transform(transform_method, **transform_kwargs)

    assert_frame_equal(gaze.frame, expected.frame)


@pytest.mark.parametrize(
    (
        'gaze_init_kwargs',
        'transform_method',
        'transform_kwargs',
        'expected_result',
        'expected_warning',
    ),
    [
        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'time': [1000],
                        'x_pix': [(100 - 1) / 2],
                        'y_pix': [0.0],
                        'distance': [1000],
                    },
                ),
                'experiment': pm.Experiment(
                    sampling_rate=1000,
                    screen_width_px=100,
                    screen_height_px=100,
                    screen_width_cm=100,
                    screen_height_cm=100,
                    distance_cm=1,
                    origin='lower left',
                ),
                'pixel_columns': ['x_pix', 'y_pix'],
                'distance_column': 'distance',
            },
            'pix2deg', {},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {
                        'time': [1000],
                        'x_pix': [49.5],
                        'y_pix': [0.0],
                        'x_dva': [0.0],
                        'y_dva': [-26.3354],
                        'distance': [1000],
                    },
                ),
                pixel_columns=['x_pix', 'y_pix'],
                position_columns=['x_dva', 'y_dva'],
                distance_column='distance',
            ),
            UserWarning,
            id='pix2deg_distance_experiment_and_distance_column_defaults_to_column',
        ),
    ],
)
def test_gaze_transfrom_expected_frame_warning(
        gaze_init_kwargs, transform_method, transform_kwargs, expected_result, expected_warning,
):
    with pytest.warns(expected_warning):
        gaze = pm.GazeDataFrame(**gaze_init_kwargs)
        gaze.transform(transform_method, **transform_kwargs)

        assert_frame_equal(gaze.frame, expected_result.frame)


@pytest.mark.parametrize(
    ('init_kwargs', 'exception', 'exception_msg'),
    [
        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'pixel_columns': 1,
            },
            TypeError,
            'pixel_columns must be of type list, but is of type int',
            id='pixel_columns_int',
        ),
    ],
)
def test_gaze_transform_raises_exception(init_kwargs, exception, exception_msg):
    with pytest.raises(exception) as excinfo:
        pm.GazeDataFrame(**init_kwargs)

    msg, = excinfo.value.args
    assert msg == exception_msg


@pytest.mark.parametrize(
    ('data', 'pixel_columns'),
    [
        pytest.param(
            {'x': [0.0], 'y': [0.0]},
            ['x', 'y'],
            id='two_components',
        ),
        pytest.param(
            {'t': [0], 'x': [0.0], 'y': [0.0]},
            ['x', 'y'],
            id='two_components_and_time',
        ),
        pytest.param(
            {'xl': [0.0], 'yl': [0.0], 'xr': [0.0], 'yr': [0.0]},
            ['xl', 'yl', 'xr', 'yr'],
            id='four_components',
        ),
        pytest.param(
            {'xl': [0.0], 'yl': [0.0], 'xr': [0.0], 'yr': [0.0], 'xa': [0.0], 'ya': [0.0]},
            ['xl', 'yl', 'xr', 'yr', 'xa', 'ya'],
            id='six_components',
        ),
    ],
)
def test_gaze_dataframe_pix2deg_creates_position_column(data, experiment, pixel_columns):
    gaze = pm.GazeDataFrame(
        data=pl.from_dict(data),
        experiment=experiment,
        pixel_columns=pixel_columns,
    )
    gaze.pix2deg()
    assert 'position' in gaze.columns


@pytest.mark.parametrize(
    ('init_kwargs', 'exception', 'expected_msg'),
    [
        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'experiment': pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
            },
            AttributeError,
            'n_components must be either 2, 4 or 6 but is None',
            id='no_column_components',
        ),
        pytest.param(
            {
                'data': pl.from_dict({'x': [0.1], 'y': [0.2]}),
                'experiment': pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
                'acceleration_columns': ['x', 'y'],
            },
            pl.exceptions.ColumnNotFoundError,
            (
                "Neither is 'pixel' in the dataframe columns, "
                'nor is a pixel column explicitly specified. '
                'You can specify the pixel column via: '
                'pix2deg(pixel_column="name_of_your_pixel_column"). '
                "Available dataframe columns are: ['time', 'acceleration']"
            ),
            id='no_pixel_column',
        ),
        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'pixel_columns': ['x', 'y'],
            },
            AttributeError,
            'experiment must not be None for this method to work',
            id='no_experiment',
        ),
        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'experiment': pm.Experiment(1024, 768, 38, 30, None, 'center', 1000),
                'pixel_columns': ['x', 'y'],
            },
            AttributeError,
            'Neither eye-to-screen distance is in the columns of the dataframe '
            'nor experiment eye-to-screen distance is specified.',
            id='no_distance_column_no_experiment_distance',
        ),
        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'experiment': pm.Experiment(1024, 768, 38, 30, None, 'center', 1000),
                'pixel_columns': ['x', 'y'],
                'distance_column': 'distance',
            },
            AttributeError,
            'Neither eye-to-screen distance is in the columns of the dataframe '
            'nor experiment eye-to-screen distance is specified.',
            id='distance_column_not_in_dataframe',
        ),
    ],
)
def test_gaze_dataframe_pix2deg_exceptions(init_kwargs, exception, expected_msg):
    gaze_df = pm.GazeDataFrame(**init_kwargs)

    with pytest.raises(exception) as excinfo:
        gaze_df.pix2deg()

    msg, = excinfo.value.args
    assert msg == expected_msg


@pytest.mark.parametrize(
    ('init_kwargs', 'warning', 'expected_msg'),
    [
        pytest.param(
            {
                'data': pl.DataFrame(schema={'d': pl.Float64, 'x': pl.Float64, 'y': pl.Float64}),
                'experiment': pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
                'pixel_columns': ['x', 'y'],
                'distance_column': 'd',
            },
            UserWarning,
            "Both a distance column and experiment's "
            'eye-to-screen distance are specified. '
            'Using eye-to-screen distances from column '
            "'distance' in the dataframe.",
            id='both_distance_column_and_experiment_distance',
        ),
    ],
)
def test_gaze_dataframe_pix2deg_warnings(init_kwargs, warning, expected_msg):
    gaze_df = pm.GazeDataFrame(**init_kwargs)

    with pytest.warns(warning) as excinfo:
        gaze_df.pix2deg()

    msg = excinfo[0].message.args[0]
    assert msg == expected_msg


@pytest.mark.parametrize(
    ('data', 'position_columns'),
    [
        pytest.param(
            {'x': [0.0], 'y': [0.0]},
            ['x', 'y'],
            id='two_components',
        ),
        pytest.param(
            {'t': [0], 'x': [0.0], 'y': [0.0]},
            ['x', 'y'],
            id='two_components_and_time',
        ),
        pytest.param(
            {'xl': [0.0], 'yl': [0.0], 'xr': [0.0], 'yr': [0.0]},
            ['xl', 'yl', 'xr', 'yr'],
            id='four_components',
        ),
        pytest.param(
            {'xl': [0.0], 'yl': [0.0], 'xr': [0.0], 'yr': [0.0], 'xa': [0.0], 'ya': [0.0]},
            ['xl', 'yl', 'xr', 'yr', 'xa', 'ya'],
            id='six_components',
        ),
    ],
)
def test_gaze_dataframe_pos2acc_creates_acceleration_column(data, experiment, position_columns):
    gaze = pm.GazeDataFrame(
        data=pl.from_dict(data),
        experiment=experiment,
        position_columns=position_columns,
    )
    gaze.pos2acc()
    assert 'acceleration' in gaze.columns


@pytest.mark.parametrize(
    ('init_kwargs', 'exception', 'expected_msg'),
    [
        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'experiment': pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
            },
            AttributeError,
            'n_components must be either 2, 4 or 6 but is None',
            id='no_column_components',
        ),
        pytest.param(
            {
                'data': pl.from_dict({'x': [0.1], 'y': [0.2]}),
                'experiment': pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
                'pixel_columns': ['x', 'y'],
            },
            pl.exceptions.ColumnNotFoundError,
            (
                "Neither is 'position' in the dataframe columns, "
                'nor is a position column explicitly specified. '
                "Since the dataframe has a 'pixel' column, "
                'consider running pix2deg() before pos2acc(). If you want to run transformations '
                "in pixel units, you can do so by using pos2acc(position_column='pixel'). "
                "Available dataframe columns are: ['time', 'pixel']"
            ),
            id='no_position_column_but_has_pixel_column',
        ),
        pytest.param(
            {
                'data': pl.from_dict({'x': [0.1], 'y': [0.2]}),
                'experiment': pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
                'acceleration_columns': ['x', 'y'],
            },
            pl.exceptions.ColumnNotFoundError,
            (
                "Neither is 'position' in the dataframe columns, "
                'nor is a position column explicitly specified. '
                'You can specify the position column via: '
                'pos2acc(position_column="your_position_column"). '
                "Available dataframe columns are: ['time', 'acceleration']"
            ),
            id='no_position_column',
        ),
        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'position_columns': ['x', 'y'],
            },
            AttributeError,
            'experiment must not be None for this method to work',
            id='no_experiment',
        ),
    ],
)
def test_gaze_dataframe_pos2acc_exceptions(init_kwargs, exception, expected_msg):
    gaze_df = pm.GazeDataFrame(**init_kwargs)

    with pytest.raises(exception) as excinfo:
        gaze_df.pos2acc()

    msg, = excinfo.value.args
    assert msg == expected_msg


@pytest.mark.parametrize(
    ('data', 'position_columns'),
    [
        pytest.param(
            {'x': [0.0], 'y': [0.0]},
            ['x', 'y'],
            id='two_components',
        ),
        pytest.param(
            {'t': [0], 'x': [0.0], 'y': [0.0]},
            ['x', 'y'],
            id='two_components_and_time',
        ),
        pytest.param(
            {'xl': [0.0], 'yl': [0.0], 'xr': [0.0], 'yr': [0.0]},
            ['xl', 'yl', 'xr', 'yr'],
            id='four_components',
        ),
        pytest.param(
            {'xl': [0.0], 'yl': [0.0], 'xr': [0.0], 'yr': [0.0], 'xa': [0.0], 'ya': [0.0]},
            ['xl', 'yl', 'xr', 'yr', 'xa', 'ya'],
            id='six_components',
        ),
    ],
)
def test_gaze_dataframe_pos2vel_creates_velocity_column(data, experiment, position_columns):
    gaze = pm.GazeDataFrame(
        data=pl.from_dict(data),
        experiment=experiment,
        position_columns=position_columns,
    )
    gaze.pos2vel(method='savitzky_golay', window_length=7, degree=2)
    assert 'velocity' in gaze.columns


@pytest.mark.parametrize(
    ('init_kwargs', 'exception', 'expected_msg'),
    [
        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'experiment': pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
            },
            AttributeError,
            'n_components must be either 2, 4 or 6 but is None',
            id='no_column_components',
        ),
        pytest.param(
            {
                'data': pl.from_dict({'x': [0.1], 'y': [0.2]}),
                'experiment': pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
                'pixel_columns': ['x', 'y'],
            },
            pl.exceptions.ColumnNotFoundError,
            (
                "Neither is 'position' in the dataframe columns, "
                'nor is a position column explicitly specified. '
                "Since the dataframe has a 'pixel' column, "
                'consider running pix2deg() before pos2vel(). If you want to run transformations '
                "in pixel units, you can do so by using pos2vel(position_column='pixel'). "
                "Available dataframe columns are: ['time', 'pixel']"
            ),
            id='no_position_column_but_has_pixel_column',
        ),
        pytest.param(
            {
                'data': pl.from_dict({'x': [0.1], 'y': [0.2]}),
                'experiment': pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
                'acceleration_columns': ['x', 'y'],
            },
            pl.exceptions.ColumnNotFoundError,
            (
                "Neither is 'position' in the dataframe columns, "
                'nor is a position column explicitly specified. '
                'You can specify the position column via: '
                'pos2vel(position_column="your_position_column"). '
                "Available dataframe columns are: ['time', 'acceleration']"
            ),
            id='no_position_column',
        ),
        pytest.param(
            {
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'position_columns': ['x', 'y'],
            },
            AttributeError,
            'experiment must not be None for this method to work',
            id='no_experiment',
        ),
    ],
)
def test_gaze_dataframe_pos2vel_exceptions(init_kwargs, exception, expected_msg):
    gaze_df = pm.GazeDataFrame(**init_kwargs)

    with pytest.raises(exception) as excinfo:
        gaze_df.pos2vel()

    msg, = excinfo.value.args
    assert msg == expected_msg


@pytest.mark.parametrize(
    ('gaze_init_kwargs', 'kwargs', 'expected'),
    [
        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'x_pix': [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                        'y_pix': [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                        'x_dva': [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                        'y_dva': [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                    },
                ),
                'pixel_columns': ['x_pix', 'y_pix'],
                'position_columns': ['x_dva', 'y_dva'],
            },
            {'method': 'moving_average', 'column': 'pixel', 'window_length': 3},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {

                        'x_pix': [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3],
                        'y_pix': [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3],
                        'x_dva': [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                        'y_dva': [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                    },
                ),
                pixel_columns=['x_pix', 'y_pix'],
                position_columns=['x_dva', 'y_dva'],
            ),
            id='pixel',
        ),
        pytest.param(
            {
                'data': pl.from_dict(
                    {
                        'x_pix': [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                        'y_pix': [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                        'x_dva': [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                        'y_dva': [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                    },
                ),
                'pixel_columns': ['x_pix', 'y_pix'],
                'position_columns': ['x_dva', 'y_dva'],
            },
            {'method': 'moving_average', 'column': 'position', 'window_length': 3},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {

                        'x_pix': [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                        'y_pix': [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                        'x_dva': [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3],
                        'y_dva': [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3],
                    },
                ),
                pixel_columns=['x_pix', 'y_pix'],
                position_columns=['x_dva', 'y_dva'],
            ),
            id='position',
        ),
    ],
)
def test_gaze_dataframe_smooth_expected_column(
        gaze_init_kwargs, kwargs, expected,
):
    gaze = pm.GazeDataFrame(**gaze_init_kwargs)
    gaze.smooth(**kwargs)

    assert_frame_equal(gaze.frame, expected.frame)
