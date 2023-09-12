# Copyright (c) 2022-2023 The pymovements Project Authors
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
"""Test pymovements.gaze.transforms.smooth."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm

@pytest.mark.parametrize(
    "kwargs, series, expected_df",
    [
        (
            {
                "method": "moving_average",
                "n_components":  1,
                "window_length": 3,
                "padding": 'wrap',

            },
            pl.Series('position', [[1.,1.],[2.,2.],[3.,3.]], pl.List(pl.Float64)),
            pl.Series('position', [[1,1],[2,2],[3, 3]], pl.List(pl.Float64)),
        )
    ]
)
def test_smooth_returns(kwargs, series, expected_df):
    """Test if smooth returns the expected dataframe."""
    df = series.to_frame()

    result_df = df.select(
        pm.gaze.transforms.smooth(**kwargs),
    )

    assert_frame_equal(result_df, expected_df.to_frame())
