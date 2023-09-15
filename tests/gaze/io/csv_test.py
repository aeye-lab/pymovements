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
"""Test read from csv."""
import pytest

import pymovements as pm


@pytest.mark.parametrize(
    ('kwargs', 'shape'),
    [
        pytest.param(
            {'file': 'tests/gaze/io/files/monocular_example.csv',
                'time_column': 'time', 'pixel_columns': ['x_left_pix', 'y_left_pix']},
            (10, 2),
            id='csv_mono_shape',
        ),
        pytest.param(
            {'file': 'tests/gaze/io/files/binocular_example.csv',
                'time_column': 'time',
                'pixel_columns': ['x_left_pix', 'y_left_pix', 'x_right_pix', 'y_right_pix'],
                'position_columns': ['x_left_pos', 'y_left_pos', 'x_right_pos', 'y_right_pos']},
            (10, 3),
            id='csv_bino_shape',
        ),
    ]
)
def test_shapes(kwargs, shape):
    gaze_dataframe = pm.gaze.from_csv(**kwargs)

    assert gaze_dataframe.frame.shape == shape
