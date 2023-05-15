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
"""Test pymovements.gaze.transforms.pix2deg"""
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {
                'screen_cm': 100,
                'distance_cm': 100,
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            TypeError,
            ('screen_px', 'missing'),
            id='no_screen_px_raises_type_error',
        ),
        pytest.param(
            {
                'screen_px': 100, 'distance_cm': 100, 'origin': 'center',
                'pixel_column': 'pixel', 'position_column': 'position',
            },
            TypeError,
            ('screen_cm', 'missing'),
            id='no_screen_cm_raises_type_error',
        ),
        pytest.param(
            {
                'screen_px': 100, 'screen_cm': 100, 'origin': 'center',
                'pixel_column': 'pixel', 'position_column': 'position',
            },
            TypeError,
            ('distance_cm', 'missing'),
            id='no_distance_cm_raises_type_error',
        ),
        pytest.param(
            {
                'screen_px': 100, 'screen_cm': 100, 'distance_cm': 100,
                'pixel_column': 'pixel', 'position_column': 'position',
            },
            TypeError,
            ('origin', 'missing'),
            id='no_origin_raises_type_error',
        ),
        pytest.param(
            {
                'screen_px': None,
                'screen_cm': 100,
                'distance_cm': 100,
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            TypeError,
            ('screen_px', 'None', 'float', 'int'),
            id='none_screen_px_raises_type_error',
        ),
        pytest.param(
            {
                'screen_px': 100,
                'screen_cm': None,
                'distance_cm': 100,
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            TypeError,
            ('screen_cm', 'None', 'float', 'int'),
            id='none_screen_cm_raises_type_error',
        ),
        pytest.param(
            {
                'screen_px': 100,
                'screen_cm': 100,
                'distance_cm': None,
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            TypeError,
            ('distance_cm', 'None', 'float', 'int'),
            id='none_distance_cm_raises_type_error',
        ),
        pytest.param(
            {
                'screen_px': 0,
                'screen_cm': 100,
                'distance_cm': 100,
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            ValueError,
            ('screen_px', 'must be greater than zero', '0'),
            id='zero_screen_px_raises_type_error',
        ),
        pytest.param(
            {
                'screen_px': 100, 'screen_cm': 0, 'distance_cm': 100, 'origin': 'center',
                'pixel_column': 'pixel', 'position_column': 'position',
            },
            ValueError,
            ('screen_cm', 'must be greater than zero', '0'),
            id='zero_screen_cm_raises_type_error',
        ),
        pytest.param(
            {
                'screen_px': 100, 'screen_cm': 100, 'distance_cm': 0, 'origin': 'center',
                'pixel_column': 'pixel', 'position_column': 'position',
            },
            ValueError,
            ('distance_cm', 'must be greater than zero', '0'),
            id='zero_distance_cm_raises_type_error',
        ),
        pytest.param(
            {
                'screen_px': -1,
                'screen_cm': 100,
                'distance_cm': 100,
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            ValueError,
            ('screen_px', 'must be greater than zero', '-1'),
            id='negative_screen_px_raises_type_error',
        ),
        pytest.param(
            {
                'screen_px': 100,
                'screen_cm': -1,
                'distance_cm': 100,
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            ValueError,
            ('screen_cm', 'must be greater than zero', '-1'),
            id='negative_screen_cm_raises_type_error',
        ),
        pytest.param(
            {
                'screen_px': 100,
                'screen_cm': 100,
                'distance_cm': -1,
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            ValueError,
            ('distance_cm', 'must be greater than zero', '-1'),
            id='negative_distance_cm_raises_type_error',
        ),
        pytest.param(
            {
                'screen_px': 100,
                'screen_cm': 100,
                'distance_cm': 100,
                'origin': 'foobar',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            ValueError,
            ('origin', 'invalid', 'foobar', 'valid', 'center', 'lower left'),
            id='invalid_origin_raises_value_error',
        ),
    ],
)
def test_pix2deg_init_raises_error(kwargs, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        pm.gaze.transforms_pl.pix2deg(**kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('kwargs', 'series', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {
                'screen_px': 100, 'screen_cm': 100, 'distance_cm': 100, 'origin': 'center',
                'pixel_column': 'aaa',
                'position_column': 'bbb',
            },
            pl.Series('ccc', [0], pl.Float64),
            pl.exceptions.ColumnNotFoundError,
            ('aaa',),
            id='df_missing_column_raises_column_not_found_error',
        ),
    ],
)
def test_pix2deg_raises_error(kwargs, series, exception, msg_substrings):
    df = series.to_frame()

    with pytest.raises(exception) as excinfo:
        df.with_columns(
            pm.gaze.transforms_pl.pix2deg(**kwargs),
        )

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    ('kwargs', 'series', 'expected_df'),
    [
        pytest.param(
            {
                'screen_px': 100,
                'screen_cm': 100,
                'distance_cm': 100,
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            pl.Series('pixel', [], pl.Float64),
            pl.Series('position', [], pl.Float64),
            id='empty_series_returns_empty_series',
        ),
        pytest.param(
            {
                'screen_px': 100,
                'screen_cm': 100,
                'distance_cm': 100,
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            pl.Series('pixel', [0], pl.Float64),
            pl.Series('position', [0], pl.Float64),
            id='zero_origin_center_returns_0',
        ),
        pytest.param(
            {
                'screen_px': 100,
                'screen_cm': 100,
                'distance_cm': 100,
                'origin': 'lower left',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            pl.Series('pixel', [(100 - 1) / 2], pl.Float64),
            pl.Series('position', [0], pl.Float64),
            id='center_pixel_origin_lowerleft_returns_0',
        ),
        pytest.param(
            {
                'screen_px': 100,
                'screen_cm': 100,
                'distance_cm': 50,
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            pl.Series('pixel', [50], pl.Float64),
            pl.Series('position', [45], pl.Float64),
            id='isosceles_triangle_origin_center_returns_45',
        ),
        pytest.param(
            {
                'screen_px': 100,
                'screen_cm': 100,
                'distance_cm': 50,
                'origin': 'lower left',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            pl.Series('pixel', [100 - 0.5], pl.Float64),
            pl.Series('position', [45], pl.Float64),
            id='isosceles_triangle_origin_lowerleft_returns_45',
        ),
        pytest.param(
            {
                'screen_px': 100,
                'screen_cm': 100,
                'distance_cm': 50,
                'origin': 'lower left',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            pl.Series('pixel', [-0.5], pl.Float64),
            pl.Series('position', [-45], pl.Float64),
            id='isosceles_triangle_left_origin_lowerleft_returns_neg45',
        ),
        pytest.param(
            {
                'screen_px': 100,
                'screen_cm': 100,
                'distance_cm': 100,
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            pl.Series('pixel', [50], pl.Float64),
            pl.Series('position', [26.565], pl.Float64),
            id='ankathet_half_origin_center_returns_26.565',
        ),
        pytest.param(
            {
                'screen_px': 100,
                'screen_cm': 100,
                'distance_cm': 100,
                'origin': 'lower left',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            pl.Series('pixel', [100 - 0.5], pl.Float64),
            pl.Series('position', [26.565], pl.Float64),
            id='ankathet_half_origin_lowerleft_returns_26.565',
        ),
        pytest.param(
            {
                'screen_px': 100,
                'screen_cm': 100,
                'distance_cm': 50 / np.sqrt(3),
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            pl.Series('pixel', [50], pl.Float64),
            pl.Series('position', [60], pl.Float64),
            id='ankathet_sqrt3_origin_center_returns_60',
        ),
        pytest.param(
            {
                'screen_px': 100,
                'screen_cm': 100,
                'distance_cm': 50 / np.sqrt(3),
                'origin': 'lower left',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            pl.Series('pixel', [100 - 0.5], pl.Float64),
            pl.Series('position', [60], pl.Float64),
            id='ankathet_sqrt3_origin_lowerleft_returns_60',
        ),
        pytest.param(
            {
                'screen_px': 100,
                'screen_cm': 100,
                'distance_cm': 50 * np.sqrt(3),
                'origin': 'center',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            pl.Series('pixel', [50], pl.Float64),
            pl.Series('position', [30], pl.Float64),
            id='opposite_sqrt3_origin_center_returns_30',
        ),
        pytest.param(
            {
                'screen_px': 100,
                'screen_cm': 100,
                'distance_cm': 50 * np.sqrt(3),
                'origin': 'lower left',
                'pixel_column': 'pixel',
                'position_column': 'position',
            },
            pl.Series('pixel', [100 - 0.5], pl.Float64),
            pl.Series('position', [30], pl.Float64),
            id='opposite_sqrt3_origin_lowerleft_returns_30',
        ),
    ],
)
def test_pix2deg_returns(kwargs, series, expected_df):
    df = series.to_frame()

    result_df = df.select(
        pm.gaze.transforms_pl.pix2deg(**kwargs),
    )
    assert_frame_equal(result_df, expected_df.to_frame())


def test_pix2deg_helper():
    s = [[0]]
    distance_cm = 100
    expected = [0.0]

    result = pm.gaze.transforms_pl.pix2deg_helper(s, distance_cm)
    assert all(result == expected)
