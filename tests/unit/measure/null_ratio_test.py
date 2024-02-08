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
"""Test null ratio measure."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


@pytest.mark.parametrize(
    ('df', 'kwargs', 'expected'),
    [
        pytest.param(
            pl.from_dict(data={'A': [[0.1, 0.2], [0.3, 0.4]],}),
            {'column': 'A'},
            pl.from_dict(data={'null_ratio': [0.0]}),
            id='list_dtype_2_elem_2_rows_no_nulls',
        ),

        pytest.param(
            pl.from_dict(data={'A': [[None, None], [None, None]],}),
            {'column': 'A'},
            pl.from_dict(data={'null_ratio': [1.0]}),
            id='list_dtype_2_elem_2_rows_all_nulls',
        ),

        pytest.param(
            pl.from_dict(data={'A': [[0.1, 0.2], [None, None]],}),
            {'column': 'A'},
            pl.from_dict(data={'null_ratio': [0.5]}),
            id='list_dtype_2_elem_2_rows_half_nulls',
        ),

    ]
)
def test_get_measure(df, kwargs, expected):
    result = df.select(pm.measure.null_ratio(**kwargs))
    assert_frame_equal(result, expected)
