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
    'kwargs, series, expected_df',
    [
        pytest.param(
            {
                'method': 'moving_average',
                'n_components': 2,
                'window_length': 1,
            },
            pl.Series('position', [[1., 1.], [2., 2.], [3., 3.]], pl.List(pl.Float64)),
            pl.Series('position', [[1., 1.], [2., 2.], [3., 3.]], pl.List(pl.Float64)),
            id='moving_average_window_length_1_returns_same_series',
        ),
        pytest.param(
            {
                'method': 'moving_average',
                'n_components': 2,
                'window_length': 3,
                'padding': None,
            },
            pl.Series('position', [[1., 1.], [2., 2.], [3., 3.]], pl.List(pl.Float64)),
            pl.Series('position', [[None, None], [2., 2.], [None, None]], pl.List(pl.Float64)),
            id='moving_average_window_length_3_no_padding',
        ),
        pytest.param(
            {
                'method': 'moving_average',
                'n_components': 2,
                'window_length': 3,
                'padding': 0.0,
            },
            pl.Series('position', [[1., 1.], [2., 2.], [3., 3.]], pl.List(pl.Float64)),
            pl.Series('position', [[1., 1.], [2., 2.], [5 / 3, 5 / 3]], pl.List(pl.Float64)),
            id='moving_average_window_length_3_constant_padding',
        ),
        pytest.param(
            {
                'method': 'moving_average',
                'n_components': 2,
                'window_length': 3,
                'padding': 'nearest',
            },
            pl.Series('position', [[1., 1.], [2., 2.], [3., 3.]], pl.List(pl.Float64)),
            pl.Series('position', [[4 / 3, 4 / 3], [2., 2.], [8 / 3, 8 / 3]], pl.List(pl.Float64)),
            id='moving_average_window_length_3_nearest_padding',
        ),
        pytest.param(
            {
                'method': 'moving_average',
                'n_components': 2,
                'window_length': 3,
                'padding': 'mirror',
            },
            pl.Series('position', [[1., 1.], [2., 2.], [3., 3.]], pl.List(pl.Float64)),
            pl.Series('position', [[5 / 3, 5 / 3], [2., 2.], [7 / 3, 7 / 3]], pl.List(pl.Float64)),
            id='moving_average_window_length_3_mirror_padding',
        ),
        pytest.param(
            {
                'method': 'exponential_moving_average',
                'n_components': 2,
                'window_length': 1,
            },
            pl.Series('position', [[1., 1.], [2., 2.], [3., 3.]], pl.List(pl.Float64)),
            pl.Series('position', [[1., 1.], [2., 2.], [3., 3.]], pl.List(pl.Float64)),
            id='exponential_moving_average_window_length_1_returns_same_series',
        ),
        pytest.param(
            {
                'method': 'savitzky_golay',
                'n_components': 2,
                'window_length': 2,
                'degree': 1,
            },
            pl.Series('position', [[1., 1.], [2., 2.], [3., 3.]], pl.List(pl.Float64)),
            pl.Series('position', [[1.5, 1.5], [2.5, 2.5], [3., 3.]], pl.List(pl.Float64)),
            id='savitzky_golay_window_length_2_degree_1_returns',
        ),
    ],
)
def test_smooth_returns(kwargs, series, expected_df):
    """Test if smooth returns the expected dataframe."""
    df = series.to_frame()

    result_df = df.select(
        pm.gaze.transforms.smooth(**kwargs),
    )

    assert_frame_equal(result_df, expected_df.to_frame())


@pytest.mark.parametrize(
    'kwargs, exception, msg_substrings',
    [
        pytest.param(
            {
                'method': 'invalid_method',
                'n_components': 2,
                'window_length': 3,
            },
            ValueError,
            "Unkown method 'invalid_method'. Supported methods are: ",
            id='invalid_method_raises_value_error',
        ),
        pytest.param(
            {
                'method': 'savitzky_golay',
                'n_components': 2,
                'window_length': 3,
                'degree': None,
            },
            TypeError,
            "'degree' must not be none for method 'savitzky_golay'",
            id='savitzky_golay_degree_none_raises_type_error',
        ),
    ],
)
def test_smooth_init_raises_error(kwargs, exception, msg_substrings):
    """Test if smooth init raises the expected error."""
    with pytest.raises(exception) as excinfo:
        pm.gaze.transforms.smooth(**kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


def test_identity_returns_same_series():
    """Test if identity returns the same series."""
    series = pl.Series('position', [[1., 1.], [2., 2.], [3., 3.]], pl.List(pl.Float64))
    series_identity = pm.gaze.transforms._identity(series)

    assert_frame_equal(series.to_frame(), series_identity.to_frame())
