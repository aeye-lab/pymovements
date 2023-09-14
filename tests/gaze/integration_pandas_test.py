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
"""Test from gaze.from_pandas."""
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


def test_from_pandas():
    pandas_df = pd.DataFrame(
        {
            'x_pix': [0, 1, 2, 3],
            'y_pix': [0, 1, 2, 3],
            'x_pos': [0, 1, 2, 3],
            'y_pos': [0, 1, 2, 3],
        },
    )

    experiment = pm.Experiment(1280, 1024, 38, 30, 68, 'lower left', 1000.0)

    gaze = pm.gaze.from_pandas(
        data=pandas_df,
        experiment=experiment,
    )

    assert gaze.frame.shape == (4, 4)
    assert all(gaze.columns == pandas_df.columns)


def test_from_pandas_explicit_columns():
    pandas_df = pd.DataFrame(
        {
            'x_pix': [0, 1, 2, 3],
            'y_pix': [4, 5, 6, 7],
            'x_pos': [9, 8, 7, 6],
            'y_pos': [5, 4, 3, 2],
        },
    )

    experiment = pm.Experiment(1280, 1024, 38, 30, 68, 'lower left', 1000.0)

    gaze = pm.gaze.from_pandas(
        data=pandas_df,
        experiment=experiment,
        pixel_columns=['x_pix', 'y_pix'],
        position_columns=['x_pos', 'y_pos'],
    )

    expected = pl.DataFrame({
        'pixel': [[0, 4], [1, 5], [2, 6], [3, 7]],
        'position': [[9, 5], [8, 4], [7, 3], [6, 2]],
    })

    assert_frame_equal(gaze.frame, expected)


@pytest.mark.parametrize(
    ('df', 'events'),
    [
        pytest.param(
            pd.DataFrame(),
            None,
            id='events_none',
        ),

        pytest.param(
            pd.DataFrame(),
            pm.EventDataFrame(),
            id='events_empty',
        ),

        pytest.param(
            pd.DataFrame(),
            pm.EventDataFrame(name='fixation', onsets=[123], offsets=[345]),
            id='fixation',
        ),

        pytest.param(
            pd.DataFrame(),
            pm.EventDataFrame(name='saccade', onsets=[34123], offsets=[67345]),
            id='saccade',
        ),

    ],
)
def test_from_pandas_events(df, events):
    if events is None:
        expected_events = pm.EventDataFrame().frame
    else:
        expected_events = events.frame

    gaze = pm.gaze.from_pandas(data=df, events=events)

    assert_frame_equal(gaze.events.frame, expected_events)
    # We don't want the events point to the same reference.
    assert gaze.events.frame is not expected_events
