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
"""Test GazeDataFrame get_measure method."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.mark.parametrize(
    ('gaze', 'method', 'kwargs', 'expected'),
    [
        pytest.param(
            pm.GazeDataFrame(
                data=pl.from_dict(data={'A': [1000, 1001, 1002, 1003]}, schema={'A': pl.Int64}),
            ),
            'null_ratio',
            {'column': 'A'},
            pl.DataFrame(data={'null_ratio': [0.0]}),
            id='null_ratio_int_column_no_nulls',
        ),

        pytest.param(
            pm.GazeDataFrame(
                data=pl.from_dict(data={'A': [1000, None, None, 1003]}, schema={'A': pl.Int64}),
            ),
            'null_ratio',
            {'column': 'A'},
            pl.DataFrame(data={'null_ratio': [0.5]}),
            id='null_ratio_int_column_half_nulls',
        ),

        pytest.param(
            pm.GazeDataFrame(
                data=pl.from_dict(data={'A': [None, None, None, None]}, schema={'A': pl.Int64}),
            ),
            'null_ratio',
            {'column': 'A'},
            pl.DataFrame(data={'null_ratio': [1.0]}),
            id='null_ratio_int_column_all_nulls',
        ),

        pytest.param(
            pm.GazeDataFrame(
                data=pl.from_dict(data={'A': [0.0, 0.1, 0.2, 0.3]}, schema={'A': pl.Float64}),
            ),
            'null_ratio',
            {'column': 'A'},
            pl.DataFrame(data={'null_ratio': [0.0]}),
            id='null_ratio_float_column_no_nulls',
        ),

        pytest.param(
            pm.GazeDataFrame(
                data=pl.from_dict(data={'A': [0.0, None, None, 0.3]}, schema={'A': pl.Float64}),
            ),
            'null_ratio',
            {'column': 'A'},
            pl.DataFrame(data={'null_ratio': [0.5]}),
            id='null_ratio_float_column_half_nulls',
        ),

        pytest.param(
            pm.GazeDataFrame(
                data=pl.from_dict(data={'A': [None, None, None, None]}, schema={'A': pl.Float64}),
            ),
            'null_ratio',
            {'column': 'A'},
            pl.DataFrame(data={'null_ratio': [1.0]}),
            id='null_ratio_float_column_all_nulls',
        ),

        pytest.param(
            pm.GazeDataFrame(
                data=pl.from_dict(data={'A': ['a', 'b', 'c', 'd']}, schema={'A': pl.Utf8}),
            ),
            'null_ratio',
            {'column': 'A'},
            pl.DataFrame(data={'null_ratio': [0.0]}),
            id='null_ratio_str_column_no_nulls',
        ),

        pytest.param(
            pm.GazeDataFrame(
                data=pl.from_dict(data={'A': ['a', None, None, 'd']}, schema={'A': pl.Utf8}),
            ),
            'null_ratio',
            {'column': 'A'},
            pl.DataFrame(data={'null_ratio': [0.5]}),
            id='null_ratio_str_column_half_nulls',
        ),

        pytest.param(
            pm.GazeDataFrame(
                data=pl.from_dict(data={'A': [None, None, None, None]}, schema={'A': pl.Utf8}),
            ),
            'null_ratio',
            {'column': 'A'},
            pl.DataFrame(data={'null_ratio': [1.0]}),
            id='null_ratio_str_column_all_nulls',
        ),

        pytest.param(
            pm.GazeDataFrame(
                data=pl.from_dict(
                    data={'t': [1000, 1001, 1002], 'x': [0.1, 0.2, 0.3], 'y': [0.1, 0.2, 0.3]},
                ),
                time_column='t',
                pixel_columns=['x', 'y'],
            ),
            'null_ratio',
            {'column': 'pixel'},
            pl.DataFrame(data={'null_ratio': [0.0]}),
            id='null_ratio_pixel_no_nulls',
        ),

        pytest.param(
            pm.GazeDataFrame(
                data=pl.from_dict(
                    data={
                        't': [1000, 1001, 1002], 'x': [None, None, None], 'y': [None, None, None],
                    },
                ),
                time_column='t',
                pixel_columns=['x', 'y'],
            ),
            'null_ratio',
            {'column': 'pixel'},
            pl.DataFrame(data={'null_ratio': [1.0]}),
            id='null_ratio_pixel_all_nulls',
        ),

        pytest.param(
            pm.GazeDataFrame(
                data=pl.from_dict(
                    data={
                        't': [1000, 1001], 'x': [0.1, None], 'y': [0.2, None],
                    },
                ),
                time_column='t',
                pixel_columns=['x', 'y'],
            ),
            'null_ratio',
            {'column': 'pixel'},
            pl.DataFrame(data={'null_ratio': [0.5]}),
            id='null_ratio_pixel_half_nulls',
        ),

        pytest.param(
            pm.GazeDataFrame(
                data=pl.from_dict(
                    data={
                        't': [1000, 1001], 'x': [0.1, None], 'y': [0.2, None],
                    },
                ),
                time_column='t',
                position_columns=['x', 'y'],
            ),
            'null_ratio',
            {'column': 'position'},
            pl.DataFrame(data={'null_ratio': [0.5]}),
            id='null_ratio_position_half_nulls',
        ),

        pytest.param(
            pm.GazeDataFrame(
                data=pl.from_dict(
                    data={
                        't': [1000, 1001], 'x': [0.1, None], 'y': [0.2, None],
                    },
                ),
                time_column='t',
                velocity_columns=['x', 'y'],
            ),
            'null_ratio',
            {'column': 'velocity'},
            pl.DataFrame(data={'null_ratio': [0.5]}),
            id='null_ratio_velocity_half_nulls',
        ),

        pytest.param(
            pm.GazeDataFrame(
                data=pl.from_dict(
                    data={'A': [1000, 1001, 1002, 1003], 'trial': [1, 1, 1, 1]},
                    schema={'A': pl.Int64, 'trial': pl.Int64},
                ),
                trial_columns='trial',
            ),
            'null_ratio',
            {'column': 'A'},
            pl.DataFrame(data={'trial': [1], 'null_ratio': [0.0]}),
            id='null_ratio_int_column_no_nulls_single_trial',
        ),

        pytest.param(
            pm.GazeDataFrame(
                data=pl.from_dict(
                    data={'A': [1000, 1001, None, None], 'trial': [1, 1, 2, 2]},
                    schema={'A': pl.Int64, 'trial': pl.Int64},
                ),
                trial_columns='trial',
            ),
            'null_ratio',
            {'column': 'A'},
            pl.DataFrame(data={'trial': [1, 2], 'null_ratio': [0.0, 1.0]}),
            id='null_ratio_int_column_no_nulls_two_trials',
        ),

    ],
)
def test_get_measure(gaze, method, kwargs, expected):
    df = gaze.get_measure(method, **kwargs)
    assert_frame_equal(df, expected)
