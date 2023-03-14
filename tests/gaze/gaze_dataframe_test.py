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
"""Test all GazeDataFrame functionality."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pymovements.gaze.experiment import Experiment
from pymovements.gaze.gaze_dataframe import GazeDataFrame


@pytest.fixture(name='experiment_fixture')
def fixture_experiment():
    return Experiment(1024, 768, 38, 30, 60, 'center', 1000)


@pytest.mark.parametrize(
    ('init_df', 'expected_velocity_columns'),
    [
        pytest.param(
            None,
            [],
            id='no_data',
        ),
        pytest.param(
            pl.DataFrame(),
            [],
            id='no_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'abc': pl.Int64}),
            [],
            id='no_velocity_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_vel': pl.Float64, 'y_vel': pl.Float64}),
            ['x_vel', 'y_vel'],
            id='no_eye_velocity_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'abc': pl.Int64, 'x_vel': pl.Float64, 'y_vel': pl.Float64}),
            ['x_vel', 'y_vel'],
            id='no_eye_velocity_columns_with_other_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_right_vel': pl.Float64, 'y_right_vel': pl.Float64}),
            ['x_right_vel', 'y_right_vel'],
            id='right_eye_velocity_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_left_vel': pl.Float64, 'y_left_vel': pl.Float64}),
            ['x_left_vel', 'y_left_vel'],
            id='left_eye_velocity_columns',
        ),
        pytest.param(
            pl.DataFrame(
                schema={
                    'x_left_vel': pl.Float64, 'y_left_vel': pl.Float64,
                    'x_right_vel': pl.Float64, 'y_right_vel': pl.Float64,
                },
            ),
            ['x_left_vel', 'y_left_vel', 'x_right_vel', 'y_right_vel'],
            id='both_eyes_velocity_columns',
        ),
    ],
)
def test_gaze_dataframe_velocity_columns(init_df, expected_velocity_columns):
    gaze_df = GazeDataFrame(init_df)

    assert set(gaze_df.velocity_columns) == set(expected_velocity_columns)


@pytest.mark.parametrize(
    ('init_df', 'expected_pixel_columns'),
    [
        pytest.param(
            pl.DataFrame(),
            [],
            id='no_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'abc': pl.Int64}),
            [],
            id='no_pix_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_pix': pl.Float64, 'y_pix': pl.Float64}),
            ['x_pix', 'y_pix'],
            id='no_eye_pix_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'abc': pl.Int64, 'x_pix': pl.Float64, 'y_pix': pl.Float64}),
            ['x_pix', 'y_pix'],
            id='no_eye_pix_pos_columns_with_other_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_right_pix': pl.Float64, 'y_right_pix': pl.Float64}),
            ['x_right_pix', 'y_right_pix'],
            id='right_eye_pix_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_left_pix': pl.Float64, 'y_left_pix': pl.Float64}),
            ['x_left_pix', 'y_left_pix'],
            id='left_eye_pix_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(
                schema={
                    'x_left_pix': pl.Float64, 'y_left_pix': pl.Float64,
                    'x_right_pix': pl.Float64, 'y_right_pix': pl.Float64,
                },
            ),
            ['x_left_pix', 'y_left_pix', 'x_right_pix', 'y_right_pix'],
            id='both_eyes_pix_pos_columns',
        ),
    ],
)
def test_gaze_dataframe_pixel_position_columns(init_df, expected_pixel_columns):
    gaze_df = GazeDataFrame(init_df)

    assert set(gaze_df.pixel_position_columns) == set(expected_pixel_columns)


@pytest.mark.parametrize(
    ('init_df', 'expected_position_columns'),
    [
        pytest.param(
            pl.DataFrame(),
            [],
            id='no_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'abc': pl.Int64}),
            [],
            id='no_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_pos': pl.Float64, 'y_pos': pl.Float64}),
            ['x_pos', 'y_pos'],
            id='no_eye_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'abc': pl.Int64, 'x_pos': pl.Float64, 'y_pos': pl.Float64}),
            ['x_pos', 'y_pos'],
            id='no_eye_pos_columns_with_other_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_right_pos': pl.Float64, 'y_right_pos': pl.Float64}),
            ['x_right_pos', 'y_right_pos'],
            id='right_eye_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_left_pos': pl.Float64, 'y_left_pos': pl.Float64}),
            ['x_left_pos', 'y_left_pos'],
            id='left_eye_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(
                schema={
                    'x_left_pos': pl.Float64, 'y_left_pos': pl.Float64,
                    'x_right_pos': pl.Float64, 'y_right_pos': pl.Float64,
                },
            ),
            ['x_left_pos', 'y_left_pos', 'x_right_pos', 'y_right_pos'],
            id='both_eyes_pos_columns',
        ),
    ],
)
def test_gaze_dataframe_position_columns(init_df, expected_position_columns):
    gaze_df = GazeDataFrame(init_df)

    assert set(gaze_df.position_columns) == set(expected_position_columns)


@pytest.mark.parametrize(
    ('init_df', 'expected_position_columns'),
    [
        pytest.param(
            pl.DataFrame(schema={'x_pix': pl.Float64, 'y_pix': pl.Float64}),
            ['x_pos', 'y_pos'],
            id='no_eye_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'abc': pl.Int64, 'x_pix': pl.Float64, 'y_pix': pl.Float64}),
            ['x_pos', 'y_pos'],
            id='no_eye_pos_columns_with_other_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_right_pix': pl.Float64, 'y_right_pix': pl.Float64}),
            ['x_right_pos', 'y_right_pos'],
            id='right_eye_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(schema={'x_left_pix': pl.Float64, 'y_left_pix': pl.Float64}),
            ['x_left_pos', 'y_left_pos'],
            id='left_eye_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(
                schema={
                    'x_left_pix': pl.Float64, 'y_left_pix': pl.Float64,
                    'x_right_pix': pl.Float64, 'y_right_pix': pl.Float64,
                },
            ),
            ['x_left_pos', 'y_left_pos', 'x_right_pos', 'y_right_pos'],
            id='both_eyes_pos_columns',
        ),
    ],
)
def test_gaze_dataframe_pix2deg_has_correct_columns(
        init_df, expected_position_columns, experiment_fixture,
):
    gaze_df = GazeDataFrame(init_df, experiment=experiment_fixture)
    gaze_df.pix2deg()

    assert set(gaze_df.position_columns) == set(expected_position_columns)


@pytest.mark.parametrize(
    ('init_kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {
                'data': pl.DataFrame(schema={'x_foo': pl.Float64, 'y_foo': pl.Float64}),
                'experiment': Experiment(1024, 768, 38, 30, 60, 'center', 1000),
            },
            AttributeError, ('pixel', 'position', 'columns', 'valid', 'x_pix', 'x_foo'),
            id='no_pix_pos_columns',
        ),
        pytest.param(
            {'data': pl.DataFrame(schema={'x_pix': pl.Float64, 'y_pix': pl.Float64})},
            AttributeError, ('experiment', 'must'),
            id='no_pix_pos_columns',
        ),
    ],
)
def test_gaze_dataframe_pix2deg_exceptions(init_kwargs, exception, msg_substrings):
    gaze_df = GazeDataFrame(**init_kwargs)

    with pytest.raises(exception) as excinfo:
        gaze_df.pix2deg()

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('init_df', 'expected_position_columns'),
    [
        pytest.param(
            pl.DataFrame(
                {'x_pix': np.arange(100), 'y_pix': np.arange(100)},
                schema={'x_pix': pl.Float64, 'y_pix': pl.Float64},
            ),
            ['x_pos', 'y_pos'],
            id='no_eye_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(
                {'abc': np.arange(100), 'x_pix': np.arange(100), 'y_pix': np.arange(100)},
                schema={'abc': pl.Int64, 'x_pix': pl.Float64, 'y_pix': pl.Float64},
            ),
            ['x_pos', 'y_pos'],
            id='no_eye_pos_columns_with_other_columns',
        ),
        pytest.param(
            pl.DataFrame(
                {'x_right_pix': np.arange(100), 'y_right_pix': np.arange(100)},
                schema={'x_right_pix': pl.Float64, 'y_right_pix': pl.Float64},
            ),
            ['x_right_pos', 'y_right_pos'],
            id='right_eye_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(
                {'x_left_pix': np.arange(100), 'y_left_pix': np.arange(100)},
                schema={'x_left_pix': pl.Float64, 'y_left_pix': pl.Float64},
            ),
            ['x_left_pos', 'y_left_pos'],
            id='left_eye_pos_columns',
        ),
        pytest.param(
            pl.DataFrame(
                {
                    'x_left_pix': np.arange(100), 'y_left_pix': np.arange(100),
                    'x_right_pix': np.arange(100), 'y_right_pix': np.arange(100),
                },
                schema={
                    'x_left_pix': pl.Float64, 'y_left_pix': pl.Float64,
                    'x_right_pix': pl.Float64, 'y_right_pix': pl.Float64,
                },
            ),
            ['x_left_pos', 'y_left_pos', 'x_right_pos', 'y_right_pos'],
            id='both_eyes_pos_columns',
        ),
    ],
)
def test_gaze_dataframe_pos2vel_has_correct_columns(
        init_df, expected_position_columns, experiment_fixture,
):
    gaze_df = GazeDataFrame(init_df, experiment=experiment_fixture)
    gaze_df.pix2deg()
    gaze_df.pos2vel()

    assert set(gaze_df.position_columns) == set(expected_position_columns)


@pytest.mark.parametrize(
    ('init_kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {
                'data': pl.DataFrame(schema={'x_pix': pl.Float64, 'y_pix': pl.Float64}),
                'experiment': Experiment(1024, 768, 38, 30, 60, 'center', 1000),
            },
            AttributeError, ('position', 'columns', 'valid', 'x_pos', 'x_pix'),
            id='no_dva_pos_columns',
        ),
        pytest.param(
            {'data': pl.DataFrame(schema={'x_pos': pl.Float64, 'y_pos': pl.Float64})},
            AttributeError, ('experiment', 'must'),
            id='no_dva_pos_columns',
        ),
    ],
)
def test_gaze_dataframe_pos2vel_exceptions(init_kwargs, exception, msg_substrings):
    gaze_df = GazeDataFrame(**init_kwargs)

    with pytest.raises(exception) as excinfo:
        gaze_df.pos2vel()

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()
