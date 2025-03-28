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
"""Test read from csv."""
import polars as pl
import pytest

from pymovements import DatasetDefinition
from pymovements import datasets
from pymovements.gaze import from_csv


@pytest.mark.parametrize(
    ('kwargs', 'expected_shape', 'expected_schema'),
    [
        pytest.param(
            {
                'file': 'tests/files/monocular_example.csv',
                'time_column': 'time',
                'time_unit': 'ms',
                'pixel_columns': ['x_left_pix', 'y_left_pix'],
            },
            (10, 2),
            {'time': pl.Int64, 'pixel': pl.List(pl.Int64)},
            id='csv_mono_shape',
        ),

        pytest.param(
            {
                'file': 'tests/files/monocular_example.csv',
                'definition': DatasetDefinition(
                    time_column='time',
                    time_unit='ms',
                    pixel_columns=['x_left_pix', 'y_left_pix'],
                ),
            },
            (10, 2),
            {'time': pl.Int64, 'pixel': pl.List(pl.Int64)},
            id='csv_mono_shape_definition',
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
            {'time': pl.Int64, 'pixel': pl.List(pl.Int64), 'position': pl.List(pl.Float64)},
            id='csv_bino_shape',
        ),

        pytest.param(
            {
                'file': 'tests/files/missing_values_example.csv',
                'time_column': 'time',
                'time_unit': 'ms',
                'pixel_columns': ['pixel_x', 'pixel_y'],
                'position_columns': ['position_x', 'position_y'],
            },
            (103, 3),
            {'time': pl.Int64, 'pixel': pl.List(pl.Float64), 'position': pl.List(pl.Float64)},
            id='csv_missing_values',
        ),

        pytest.param(
            {
                'file': 'tests/files/gaze_on_faces_example.csv',
                'definition': datasets.GazeOnFaces(),
            },
            (10, 2),
            {'time': pl.Float64, 'pixel': pl.List(pl.Float32)},
        ),

        pytest.param(
            {
                'file': 'tests/files/gaze_on_faces_example.csv',
                'definition': datasets.GazeOnFaces(),
                'pixel_columns': ['foo', 'bar'],
                **{
                    'separator': ',',
                    'has_header': False,
                    'new_columns': ['foo', 'bar'],
                    'schema_overrides': [pl.Float32, pl.Float32],
                },
            },
            (10, 2),
            {'time': pl.Float64, 'pixel': pl.List(pl.Float32)},
            id='gaze_on_faces_dataset_explicit_read_kwargs_and_columns',
        ),

        pytest.param(
            {
                'file': 'tests/files/gazebase_example.csv',
                'definition': datasets.GazeBase(),
            },
            (10, 7),
            {
                'time': pl.Int64, 'validity': pl.Int64, 'dP': pl.Float32, 'lab': pl.Int64,
                'x_target_pos': pl.Float32, 'y_target_pos': pl.Float32,
                'position': pl.List(pl.Float32),
            },
            id='gazebase_dataset_example',
        ),

        pytest.param(
            {
                'file': 'tests/files/gazebase_example.csv',
                'definition': datasets.GazeBase(),
                'column_map': {'dP': 'test'},
            },
            (10, 7),
            {
                'time': pl.Int64, 'val': pl.Int64, 'test': pl.Float32, 'lab': pl.Int64,
                'xT': pl.Float32, 'yT': pl.Float32,
                'position': pl.List(pl.Float32),
            },
            id='gazebase_dataset_example_explicit_column_map',
        ),

        pytest.param(
            {
                'file': 'tests/files/gazebase_vr_example.csv',
                'definition': datasets.GazeBaseVR(),
            },
            (10, 11),
            {
                'time': pl.Float32,
                'x_target_pos': pl.Float32, 'y_target_pos': pl.Float32, 'z_target_pos': pl.Float32,
                'clx': pl.Float32, 'cly': pl.Float32, 'clz': pl.Float32,
                'crx': pl.Float32, 'cry': pl.Float32, 'crz': pl.Float32,
                'position': pl.List(pl.Float32),
            },
            id='gazebase_vr_dataset_example',
        ),

        pytest.param(
            {
                'file': 'tests/files/hbn_example.csv',
                'definition': datasets.HBN(),
            },
            (10, 2),
            {'time': pl.Float64, 'pixel': pl.List(pl.Float32)},
            id='hbn_dataset_example',
        ),

        pytest.param(
            {
                'file': 'tests/files/hbn_example.csv',
                'definition': datasets.HBN(),
                'pixel_columns': [],
                'position_columns': ['x_pix', 'y_pix'],
            },
            (10, 2),
            {'time': pl.Float64, 'position': pl.List(pl.Float32)},
            id='hbn_dataset_example_explicit_columns',
        ),

        pytest.param(
            {
                'file': 'tests/files/judo1000_example.csv',
                'definition': datasets.JuDo1000(),
            },
            (10, 4),
            {
                'trial_id': pl.Int64, 'point_id': pl.Int64,
                'time': pl.Int64, 'pixel': pl.List(pl.Float32),
            },
            id='judo1000_dataset_example',
        ),

        pytest.param(
            {
                'file': 'tests/files/sbsat_example.csv',
                'definition': datasets.SBSAT(),
            },
            (10, 5),
            {
                'book_name': pl.String, 'screen_id': pl.Int64, 'time': pl.Int64,
                'pupil_left': pl.Float32, 'pixel': pl.List(pl.Float32),
            },
            id='sbsat_dataset_example',
        ),
    ],
)
def test_from_csv_gaze_has_expected_shape_and_columns(kwargs, expected_shape, expected_schema):
    gaze_dataframe = from_csv(**kwargs)

    assert gaze_dataframe.frame.shape == expected_shape
    assert gaze_dataframe.frame.schema == expected_schema
