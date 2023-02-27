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
"""
Test pymovements filters.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.utils.filters import filter_candidates_remove_nans, events_split_nans, filter_and_split

@pytest.mark.parametrize(
    'params, expected',
    [
        pytest.param(
            {
                'candidates': [[0,1,2,3,4],
                                [5,6,7,8]],
                'values': np.array([(np.nan, np.nan), (0, 0),
                       (0, 0), (0, 0),
                       (np.nan, np.nan),
                       (np.nan, np.nan),
                       (0, 0), (0, 0),
                       (0, 0)]),
            },
            {'values_filter':[[1, 2, 3], [6, 7, 8]]},
            id='test_filters',
        ),
        pytest.param(
            {
                'candidates': [[0,1,2,3,4,5,6,7]],
                'values': np.array([(0, 0),
                       (0, 0), (0, 0),
                       (np.nan, np.nan),
                       (np.nan, np.nan),
                       (0, 0), (0, 0),
                       (0, 0)]),
            },
            {'values_split':[[0, 1, 2], [5, 6, 7]]},
            id='test_events_split',
        ),
        pytest.param(
            {
                'candidates': [[0,1,2,3,4,5,6,7]],
                'values': np.array([(0, 0),
                       (0, 0), (0, 0),
                       (np.nan, np.nan),
                       (np.nan, np.nan),
                       (0, 0), (np.nan, np.nan),
                       (0, 0)]),
                'flag_split_events': True,
            },
            {'values_filter_split':[[0, 1, 2], [5], [7]]},
            id='test_events_filter_split',
        ),
        
    ],
)

def test_filters(params, expected):
    if 'values_filter' in expected:
        results = filter_candidates_remove_nans(params['candidates'],
                                                            params['values'])
        assert expected['values_filter'] == results
    
    if 'values_split' in expected:
        results = events_split_nans(params['candidates'],
                                                            params['values'])
        assert expected['values_split'] == results
        
    if 'values_filter_split' in expected:
        results = filter_and_split(params['candidates'],
                                                            params['values'],
                                                            params['flag_split_events'])
        assert expected['values_filter_split'] == results