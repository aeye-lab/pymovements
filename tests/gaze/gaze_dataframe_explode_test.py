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
"""Test GazeDataFrame.explode()."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.mark.parametrize(
    ('init_data', 'explode_kwargs', 'expected'),
    [
        pytest.param(
            pl.DataFrame(schema={'pixel': pl.List(pl.Float64)}),
            {'column': 'pixel', 'output_columns': ['x', 'y']},
            pl.DataFrame(schema={'x': pl.Float64, 'y': pl.Float64}),
            id='empty_df_with_schema_two_pixel_columns',
        ),

        pytest.param(
            pl.DataFrame(schema={'abc': pl.Int64, 'pixel': pl.List(pl.Float64)}),
            {'column': 'pixel', 'output_columns': ['x', 'y']},
            pl.DataFrame(schema={'abc': pl.Int64, 'x': pl.Float64, 'y': pl.Float64}),
            id='empty_df_with_three_column_schema_two_pixel_columns',
        ),

        pytest.param(
            pl.DataFrame(schema={'pixel': pl.List(pl.Float64)}),
            {'column': 'pixel', 'output_columns': ['xl', 'yl', 'xr', 'yr']},
            pl.DataFrame(
                schema={
                    'xl': pl.Float64, 'yl': pl.Float64, 'xr': pl.Float64, 'yr': pl.Float64,
                },
            ),
            id='empty_df_with_schema_four_pixel_columns',
        ),

        pytest.param(
            pl.DataFrame(schema={'pixel': pl.List(pl.Float64)}),
            {'column': 'pixel', 'output_columns': ['xl', 'yl', 'xr', 'yr', 'xa', 'ya']},
            pl.DataFrame(
                schema={
                    'xl': pl.Float64, 'yl': pl.Float64,
                    'xr': pl.Float64, 'yr': pl.Float64,
                    'xa': pl.Float64, 'ya': pl.Float64,
                },
            ),
            id='empty_df_with_schema_six_pixel_columns',
        ),

        pytest.param(
            pl.DataFrame({'pixel': [[1.23, 4.56]]}),
            {'column': 'pixel', 'output_columns': ['x', 'y']},
            pl.DataFrame({'x': [1.23], 'y': [4.56]}),
            id='df_single_row_two_pixel_columns',
        ),

        pytest.param(
            pl.DataFrame({'abc': [1], 'pixel': [[1.23, 4.56]]}),
            {'column': 'pixel', 'output_columns': ['x', 'y']},
            pl.DataFrame({'abc': [1], 'x': [1.23], 'y': [4.56]}),
            id='df_single_row_three_columns_two_pixel_columns',
        ),

        pytest.param(
            pl.DataFrame({'pixel': [[1.2, 3.4, 5.6, 7.8]]}),
            {'column': 'pixel', 'output_columns': ['xl', 'yl', 'xr', 'yr']},
            pl.DataFrame({'xl': [1.2], 'yl': [3.4], 'xr': [5.6], 'yr': [7.8]}),
            id='df_single_row_four_pixel_columns',
        ),

        pytest.param(
            pl.DataFrame({'pixel': [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]}),
            {'column': 'pixel', 'output_columns': ['xl', 'yl', 'xr', 'yr', 'xa', 'ya']},
            pl.DataFrame({'xl': [.1], 'yl': [.2], 'xr': [.3], 'yr': [.4], 'xa': [.5], 'ya': [.6]}),
            id='df_single_row_six_pixel_columns',
        ),
    ],
)
def test_gaze_dataframe_explode_has_expected_frame(init_data, explode_kwargs, expected):
    gaze = pm.GazeDataFrame(init_data)
    gaze.explode(**explode_kwargs)
    assert_frame_equal(gaze.frame, expected)
