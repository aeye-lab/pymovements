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
"""Test from gaze.from_numpy."""
import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

import pymovements as pm


def test_from_numpy():
    array = np.array(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ],
    )

    schema = ['x_pix', 'y_pix', 'x_pos', 'y_pos']

    experiment = pm.Experiment(
        screen_width_px=1280,
        screen_height_px=1024,
        screen_width_cm=38,
        screen_height_cm=30,
        distance_cm=68,
        origin='lower left',
        sampling_rate=1000.0,
    )

    gaze = pm.gaze.from_numpy(
        data=array,
        schema=schema,
        experiment=experiment,
    )

    assert gaze.frame.shape == (4, 4)
    assert gaze.columns == schema


def test_from_numpy_with_schema():
    array = np.array(
        [
            [101, 102, 103, 104],
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [9, 8, 7, 6],
            [5, 4, 3, 2],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [2, 3, 4, 5],
            [6, 7, 8, 9],
        ],
        dtype=np.float64,
    )

    schema = ['t', 'x_pix', 'y_pix', 'x_pos', 'y_pos', 'x_vel', 'y_vel', 'x_acc', 'y_acc']

    experiment = pm.Experiment(
        screen_width_px=1280,
        screen_height_px=1024,
        screen_width_cm=38,
        screen_height_cm=30,
        distance_cm=68,
        origin='lower left',
        sampling_rate=1000.0,
    )

    gaze = pm.gaze.from_numpy(
        data=array,
        schema=schema,
        experiment=experiment,
        time_column='t',
        pixel_columns=['x_pix', 'y_pix'],
        position_columns=['x_pos', 'y_pos'],
        velocity_columns=['x_vel', 'y_vel'],
        acceleration_columns=['x_acc', 'y_acc'],
    )

    expected = pl.DataFrame(
        {
            'time': [101, 102, 103, 104],
            'pixel': [[0, 4], [1, 5], [2, 6], [3, 7]],
            'position': [[9, 5], [8, 4], [7, 3], [6, 2]],
            'velocity': [[1, 5], [2, 6], [3, 7], [4, 8]],
            'acceleration': [[2, 6], [3, 7], [4, 8], [5, 9]],
        },
        schema={
            'time': pl.Float64,
            'pixel': pl.List(pl.Float64),
            'position': pl.List(pl.Float64),
            'velocity': pl.List(pl.Float64),
            'acceleration': pl.List(pl.Float64),
        },
    )

    assert_frame_equal(gaze.frame, expected)
    assert gaze.n_components == 2


def test_from_numpy_explicit_columns():
    time = np.array([101, 102, 103, 104])
    pixel = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
    position = np.array([[9, 8, 7, 6], [5, 4, 3, 2]], dtype=np.int64)
    velocity = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
    acceleration = np.array([[2, 3, 4, 5], [6, 7, 8, 9]], dtype=np.int64)

    experiment = pm.Experiment(
        screen_width_px=1280,
        screen_height_px=1024,
        screen_width_cm=38,
        screen_height_cm=30,
        distance_cm=68,
        origin='lower left',
        sampling_rate=1000.0,
    )

    gaze = pm.gaze.from_numpy(
        time=time,
        pixel=pixel,
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        experiment=experiment,
    )

    expected = pl.DataFrame({
        'time': [101, 102, 103, 104],
        'pixel': [[0, 4], [1, 5], [2, 6], [3, 7]],
        'position': [[9, 5], [8, 4], [7, 3], [6, 2]],
        'velocity': [[1, 5], [2, 6], [3, 7], [4, 8]],
        'acceleration': [[2, 6], [3, 7], [4, 8], [5, 9]],
    })

    assert_frame_equal(gaze.frame, expected)
    assert gaze.n_components == 2


def test_init_all_none():
    gaze = pm.gaze.from_numpy(
        data=None,
        schema=None,
        experiment=None,
        time=None,
        pixel=None,
        position=None,
        velocity=None,
        acceleration=None,
        time_column=None,
        pixel_columns=None,
        position_columns=None,
        velocity_columns=None,
        acceleration_columns=None,
    )

    expected = pl.DataFrame()

    assert_frame_equal(gaze.frame, expected)
    assert gaze.n_components is None
