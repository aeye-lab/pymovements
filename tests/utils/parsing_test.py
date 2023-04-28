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
""" Tests pymovements asc to csv processing"""
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


ASC_TEXT = r"""
some
lines
to
ignore
the next line has all additional trial columns set to None
10000000	  850.7	  717.5	  714.0	    0.0	...
MSG 10000001 START_A
the next line now should have the task column set to A
10000002	  850.7	  717.5	  714.0	    0.0	...
MSG 10000003 STOP_A
the task should be set to None again
10000004	  850.7	  717.5	  714.0	    0.0	...
MSG 10000005 START_B
the next line now should have the task column set to B
10000006	  850.7	  717.5	  714.0	    0.0	...
MSG 10000007 START_TRIAL_1
the next line now should have the trial column set to 1
10000008	  850.7	  717.5	  714.0	    0.0	...
MSG 10000009 STOP_TRIAL_1
MSG 10000010 START_TRIAL_2
the next line now should have the trial column set to 2
10000011	  850.7	  717.5	  714.0	    0.0	...
MSG 10000012 STOP_TRIAL_2
MSG 10000013 START_TRIAL_3
the next line now should have the trial column set to 3
10000014	  850.7	  717.5	  714.0	    0.0	...
MSG 10000015 STOP_TRIAL_3
MSG 10000016 STOP_B
task and trial should be set to None again
10000017	  850.7	  717.5	  .	    0.0	...
"""

PATTERNS = [
    {
        'pattern': 'START_A',
        'column': 'task',
        'value': 'A',
    },
    {
        'pattern': 'START_B',
        'column': 'task',
        'value': 'B',
    },
    {
        'pattern': ('STOP_A', 'STOP_B'),
        'column': 'task',
        'value': None,
    },

    r'START_TRIAL_(?P<trial_id>\d+)',
    {
        'pattern': r'STOP_TRIAL',
        'column': 'trial_id',
        'value': None,
    },
]

EXPECTED_DF = pl.from_dict(
    {
        'time': [10000000, 10000002, 10000004, 10000005, 10000008, 10000011, 10000014, 10000017],
        'x_pix': [850.7, 850.7, 850.7, 850.7, 850.7, 850.7, 850.7, 850.7],
        'y_pix': [717.5, 717.5, 717.5, 717.5, 717.5, 717.5, 717.5, 717.5],
        'pupil': [714.0, 714.0, 714.0, 714.0, 714.0, 714.0, 714.0, np.nan],
        'task': [None, 'A', None, 'B', 'B', 'B', 'B', None],
        'trial_id': [None, None, None, None, '1', '2', '3', None],
    },
)


def test_parse_eyelink(tmp_path):
    filepath = tmp_path / 'sub.asc'
    filepath.write_text(ASC_TEXT)

    df = pm.utils.parsing.parse_eyelink(
        filepath,
        patterns=PATTERNS,
    )

    assert_frame_equal(df, EXPECTED_DF, check_column_order=False)


@pytest.mark.parametrize(
    'patterns',
    [
        [1],
        [{'pattern': 1}],
    ],
)
def test_parse_eyelink_raises_value_error(tmp_path, patterns):
    filepath = tmp_path / 'sub.asc'
    filepath.write_text(ASC_TEXT)

    with pytest.raises(ValueError) as excinfo:
        pm.utils.parsing.parse_eyelink(
            filepath,
            patterns=patterns,
        )

    msg, = excinfo.value.args

    expected_substrings = ['invalid pattern', '1']
    for substring in expected_substrings:
        assert substring in msg
