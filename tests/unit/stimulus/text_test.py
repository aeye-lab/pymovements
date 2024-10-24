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
            'A',
            'B',
            'S',
            'T',
            'R',
            'A',
            'C',
            'T',
            'p',
            'y',
            'm',
            'o',
        ],
        'top_left_x': [
            400.0,
            415.0,
            430.0,
            445.0,
            460.0,
            475.0,
            490.0,
            505.0,
            400.0,
            414.972602739726,
            429.94520547945206,
            444.9178082191781,
        ],
        'top_left_y': [
            122.0,
            122.0,
            122.0,
            122.0,
            122.0,
            122.0,
            122.0,
            122.0,
            214.85148514851485,
            214.85148514851485,
            214.85148514851485,
            214.85148514851485,
        ],
        'width': [
            15.0,
            15.0,
            15.0,
            15.0,
            15.0,
            15.0,
            15.0,
            15.0,
            14.972602739726028,
            14.972602739726028,
            14.972602739726028,
            14.972602739726028,
        ],
        'height': [
            18,
            18,
            18,
            18,
            18,
            18,
            18,
            18,
            23,
            23,
            23,
            23,
        ],
        'char_idx_in_line': [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            0,
            1,
            2,
            3,
        ],
        'line_idx': [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
        ],
        'page': ['page_2' for _ in range(12)],
        'word': [
            'ABSTRACT',
            'ABSTRACT',
            'ABSTRACT',
            'ABSTRACT',
            'ABSTRACT',
            'ABSTRACT',
            'ABSTRACT',
            'ABSTRACT',
            'pymovements:',
            'pymovements:',
            'pymovements:',
            'pymovements:',
        ],
        'bottom_left_x': [
            415.0,
            430.0,
            445.0,
            460.0,
            475.0,
            490.0,
            505.0,
            520.0,
            414.972602739726,
            429.94520547945206,
            444.9178082191781,
            459.8904109589041,
        ],
        'bottom_left_y': [
            140.0,
            140.0,
            140.0,
            140.0,
            140.0,
            140.0,
            140.0,
            140.0,
            237.85148514851485,
            237.85148514851485,
            237.85148514851485,
            237.85148514851485,
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
        start_x_column='top_left_x',
        start_y_column='top_left_y',
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
            start_x_column='top_left_x',
            start_y_column='top_left_y',
            width_column='width',
            height_column='height',
            page_column='page',
        )
    msg, = excinfo.value.args
    expected = 'unsupported file format ".pickle".Supported formats are: '\
        '[\'.csv\', \'.tsv\', \'.txt\']'
    assert msg == expected


@pytest.mark.parametrize(
    ('aoi_file', 'custom_read_kwargs'),
    [
        pytest.param(
            'tests/files/toy_text_1_1_aoi.csv',
            None,
            id='toy_text_1_1_aoi',
        ),
        pytest.param(
            Path('tests/files/toy_text_1_1_aoi.csv'),
            {'separator': ','},
            id='toy_text_1_1_aoi',
        ),
    ],
)
def test_text_stimulus_splitting(aoi_file, custom_read_kwargs):
    aois_df = pm.stimulus.text.from_file(
        aoi_file,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
        custom_read_kwargs=custom_read_kwargs,
    )

    aois_df.split_aois_by(by='line_idx')
    assert len(aois_df.aois) == 2
