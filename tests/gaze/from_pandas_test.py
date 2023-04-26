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

    experiment = pm.Experiment(
        screen_width_px=1280,
        screen_height_px=1024,
        screen_width_cm=38,
        screen_height_cm=30,
        distance_cm=68,
        origin='lower left',
        sampling_rate=1000.0,
    )

    gaze = pm.gaze.from_pandas(
        data=pandas_df,
        experiment=experiment,
    )

    assert gaze.frame.shape == (4, 4)
    assert all(gaze.columns == pandas_df.columns)
