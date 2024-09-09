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
"""Test read from csv."""
import polars as pl
import pytest

import pymovements as pm


@pytest.mark.parametrize(
    ('kwargs', 'shape'),
    [
        pytest.param(
            {
                'file': 'tests/files/monocular_example.csv',
                'time_column': 'time',
                'time_unit': 'ms',
                'pixel_columns': ['x_left_pix', 'y_left_pix'],
            },
            (10, 2),
            id='csv_mono_shape',
        ),
        pytest.param(
            {
                'file': 'tests/files/binocular_example.csv',
                'time_column': 'time',
                'time_unit': 'ms',
                'pixel_columns': ['x_left_pix', 'y_left_pix', 'x_right_pix', 'y_right_pix'],
                'position_columns': ['x_left_pos', 'y_left_pos', 'x_right_pos', 'y_right_pos'],
            },
            (10, 3),
            id='csv_bino_shape',
        ),
        pytest.param(
            {
                'file': 'tests/files/hbn_example.csv',
                'time_column': pm.datasets.HBN().time_column,
                'time_unit': pm.datasets.HBN().time_unit,
                'experiment': pm.datasets.HBN().experiment,
                'pixel_columns': pm.datasets.HBN().pixel_columns,
            },
            (10, 2),
            id='hbn_dataset_example',
        ),
        pytest.param(
            {
                'file': 'tests/files/sbsat_example.csv',
                'time_column': pm.datasets.SBSAT().time_column,
                'time_unit': pm.datasets.SBSAT().time_unit,
                'pixel_columns': pm.datasets.SBSAT().pixel_columns,
                **pm.datasets.SBSAT().custom_read_kwargs['gaze'],
            },
            (10, 5),
            id='sbsat_dataset_example',
        ),
        pytest.param(
            {
                'file': 'tests/files/gazebase_example.csv',
                'time_column': pm.datasets.GazeBase().time_column,
                'time_unit': pm.datasets.GazeBase().time_unit,
                'position_columns': pm.datasets.GazeBase().position_columns,
                **pm.datasets.GazeBase().custom_read_kwargs['gaze'],
            },
            (10, 7),
            id='gazebase_dataset_example',
        ),
        pytest.param(
            {
                'file': 'tests/files/gaze_on_faces_example.csv',
                'time_column': pm.datasets.GazeOnFaces().time_column,
                'time_unit': pm.datasets.GazeOnFaces().time_unit,
                'pixel_columns': pm.datasets.GazeOnFaces().pixel_columns,
                **pm.datasets.GazeOnFaces().custom_read_kwargs['gaze'],
            },
            (10, 1),
            id='gaze_on_faces_dataset_example',
        ),
        pytest.param(
            {
                'file': 'tests/files/gazebase_vr_example.csv',
                'time_column': pm.datasets.GazeBaseVR().time_column,
                'time_unit': pm.datasets.GazeBaseVR().time_unit,
                'position_columns': pm.datasets.GazeBaseVR().position_columns,
            },
            (10, 11),
            id='gazebase_vr_dataset_example',
        ),
        pytest.param(
            {
                'file': 'tests/files/judo1000_example.csv',
                'time_column': pm.datasets.JuDo1000().time_column,
                'time_unit': pm.datasets.JuDo1000().time_unit,
                'pixel_columns': pm.datasets.JuDo1000().pixel_columns,
                **pm.datasets.JuDo1000().custom_read_kwargs['gaze'],
            },
            (10, 4),
            id='judo1000_dataset_example',
        ),
    ],
)
def test_shapes(kwargs, shape):
    gaze_dataframe = pm.gaze.from_csv(**kwargs)
    assert gaze_dataframe.frame.shape == shape


@pytest.mark.parametrize(
    ('kwargs', 'dtypes'),
    [
        pytest.param(
            {
                'file': 'tests/files/monocular_example.csv',
                'time_column': 'time',
                'time_unit': 'ms',
                'pixel_columns': ['x_left_pix', 'y_left_pix'],
            },
            [pl.Int64, pl.List(pl.Int64)],
            id='csv_mono_dtypes',
        ),
        pytest.param(
            {
                'file': 'tests/files/missing_values_example.csv',
                'time_column': 'time',
                'time_unit': 'ms',
                'pixel_columns': ['pixel_x', 'pixel_y'],
                'position_columns': ['position_x', 'position_y'],
            },
            [pl.Int64, pl.List(pl.Float64), pl.List(pl.Float64)],
            id='csv_missing_values_dtypes',
        ),
    ],
)
def test_dtypes(kwargs, dtypes):
    gaze_dataframe = pm.gaze.from_csv(**kwargs)
    assert gaze_dataframe.frame.dtypes == dtypes
