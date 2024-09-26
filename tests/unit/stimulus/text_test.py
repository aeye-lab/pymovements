# Copyright (c) 2024 The pymovements Project Authors
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
"""Test Text stimulus class."""
from pathlib import Path

import polars
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


EXPECTED_DF = polars.DataFrame(
    {
        'char': [
            'W',
            'h',
            'a',
            't',
            ' ',
            'i',
            's',
            ' ',
            'p',
            'y',
            'm',
            'o',
        ],
        'top_left_x': [
            400.0,
            414.95,
            429.9,
            444.85,
            459.8,
            474.75,
            489.7,
            504.65,
            519.6,
            534.55,
            549.5,
            564.45,
        ],
        'top_left_y': [122.0 for _ in range(12)],
        'width': [14.95 for _ in range(12)],
        'height': [23 for _ in range(12)],
        'char_idx_in_line': list(range(12)),
        'line_idx': [0 for _ in range(12)],
        'page': ['question_1' for _ in range(12)],
        'word': [
            'What',
            'What',
            'What',
            'What',
            ' ',
            'is',
            'is',
            ' ',
            'pymovements?',
            'pymovements?',
            'pymovements?',
            'pymovements?',
        ],
    },
)


@pytest.mark.parametrize(
    ('aoi_file', 'custom_read_kwargs', 'expected'),
    [
        pytest.param(
            'tests/files/toy_text_1_1_aoi.csv',
            None,
            EXPECTED_DF,
            id='toy_text_1_1_aoi',
        ),
        pytest.param(
            Path('tests/files/toy_text_1_1_aoi.csv'),
            {'separator': ','},
            EXPECTED_DF,
            id='toy_text_1_1_aoi',
        ),
    ],
)
def test_text_stimulus(aoi_file, custom_read_kwargs, expected):
    aois = pm.stimulus.text.from_file(
        aoi_file,
        aoi_column='char',
        pixel_x_column='top_left_x',
        pixel_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
        custom_read_kwargs=custom_read_kwargs,
    )
    head = aois.aois.head(12)

    assert_frame_equal(
        head,
        expected,
    )
    assert len(aois.aois.columns) == len(expected.columns)


def test_text_stimulus_unsupported_format():
    with pytest.raises(ValueError) as excinfo:
        pm.stimulus.text.from_file(
            'tests/files/toy_text_1_1_aoi.pickle',
            aoi_column='char',
            pixel_x_column='top_left_x',
            pixel_y_column='top_left_y',
            width_column='width',
            height_column='height',
            page_column='page',
        )
    msg, = excinfo.value.args
    expected = 'unsupported file format ".pickle".Supported formats are: '\
        '[\'.csv\', \'.tsv\', \'.txt\']'
    assert msg == expected
