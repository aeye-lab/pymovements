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
                        'y_pix': [
                            20.0, 20.200000000000003, 20.400000000000006,
                            20.60000000000001, 20.80000000000001,
                        ],
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
                        'x_dva': [26.335410003881348, 26.335410003881348],
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
                        'x_dva': [26.335410003881348, 26.335410003881348],
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
                        'y_dva': [-26.335410003881348],
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
            'pos2vel', {'method': 'preceding'},
            pm.GazeDataFrame(
                data=pl.from_dict(
                    {
                        'time': [1000, 1001, 1002],
                        'x_dva': [1.0, 1.0, 1.0],
                        'y_dva': [1.0, 1.1, 1.2],
                        'x_vel': [None, 0.0, 0.0],
                        'y_vel': [None, 100.00000000000009, 99.99999999999987],
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
                        'y_vel': [None, 99.99999999999997, None],
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
                        'y_vel': [None, None, 100.00000000000001, None, None],
                    },
                ),
                position_columns=['x_dva', 'y_dva'],
                velocity_columns=['x_vel', 'y_vel'],
            ),
            id='pos2vel_five_point',
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
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'experiment': pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
                'acceleration_columns': ['x', 'y'],
            },
            pl.exceptions.ColumnNotFoundError,
            (
                'pixel\n\nError originated just after this operation:\nDF ["acceleration"]; '
                'PROJECT */1 COLUMNS; SELECTION: "None"'
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
    ],
)
def test_gaze_dataframe_pix2deg_exceptions(init_kwargs, exception, expected_msg):
    gaze_df = pm.GazeDataFrame(**init_kwargs)

    with pytest.raises(exception) as excinfo:
        gaze_df.pix2deg()

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
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'experiment': pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
                'pixel_columns': ['x', 'y'],
            },
            pl.exceptions.ColumnNotFoundError,
            (
                'position\n\nError originated just after this operation:\nDF ["pixel"]; '
                'PROJECT */1 COLUMNS; SELECTION: "None"'
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
                'data': pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
                'experiment': pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
                'pixel_columns': ['x', 'y'],
            },
            pl.exceptions.ColumnNotFoundError,
            (
                'position\n\nError originated just after this operation:\nDF ["pixel"]; '
                'PROJECT */1 COLUMNS; SELECTION: "None"'
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
