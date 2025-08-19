# Copyright (c) 2024-2025 The pymovements Project Authors
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

from pymovements.stimulus import text


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

EXPECTED_AOI_MULTIPLEYE_STIMULI_TOY_X_1_TEXT_1_1 = polars.DataFrame(
    {
        'char': [
            'W', 'h', 'a', 't', ' ', 'i', 's', ' ', 'p', 'y'
        ],
        'top_left_x': [
            81.0, 94.0, 107.0, 120.0, 133.0, 146.0, 159.0, 172.0, 185.0, 198.0
        ],
        'top_left_y': [
            99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0
        ],
        'width': [
            13, 13, 13, 13, 13, 13, 13, 13, 13, 13
        ],
        'height': [
            30, 30, 30, 30, 30, 30, 30, 30, 30, 30
        ],
        'char_idx_in_line': [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        ],
        'line_idx': [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ],
        'page': [
            'question_01111' for _ in range(10)
        ],
        'word_idx': [
            0, 0, 0, 0, 1, 1, 1, 1, 2, 2
        ],
        'word_idx_in_line': [
            0, 0, 0, 0, 1, 1, 1, 1, 2, 2
        ],
        'word': [
            'What', 'What', 'What', 'What', '', 'is', 'is', '', 'pymovements?', 'pymovements?'
        ],
        'question_image_version': [
            'question_images_version_1' for _ in range(10)
        ],
    }
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
            id='toy_text_1_1_aoi_sep',
        ),
    ],
)
def test_text_stimulus(aoi_file, custom_read_kwargs, expected):
    aois = text.from_file(
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
        text.from_file(
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
        '[\'.csv\', \'.ias\', \'.tsv\', \'.txt\']'
    assert msg == expected

@pytest.mark.parametrize(
    ('aoi_dir_path', 'custom_read_kwargs', 'expected'),
    [
        pytest.param(
            'tests/files/aoi_multipleye_stimuli_toy_x_1',
            None,
            EXPECTED_AOI_MULTIPLEYE_STIMULI_TOY_X_1_TEXT_1_1,
            id='toy_text_1_1_aoi_dir',
        ),
        pytest.param(
            Path('tests/files/toy_text_1_1_aoi.csv'),
            {'separator': ','},
            EXPECTED_AOI_MULTIPLEYE_STIMULI_TOY_X_1_TEXT_1_1,
            id='toy_text_1_1_aoi_sep_dir',
        ),
    ],
)
def test_text_stimuli_from_path(aoi_dir_path, custom_read_kwargs, expected):
    aois_list = text.from_path(
        aoi_dir_path,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
        custom_read_kwargs=custom_read_kwargs,
    )
    assert(len(aois_list) == 4)
    head = aois_list[0].aois.head(10)

    assert_frame_equal(
        head,
        expected,
    )
    assert len(aois_list[0].aois.columns) == len(expected.columns)

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
            id='toy_text_1_1_aoi_sep',
        ),
    ],
)
def test_text_stimulus_splitting(aoi_file, custom_read_kwargs):
    aois_df = text.from_file(
        aoi_file,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
        custom_read_kwargs=custom_read_kwargs,
    )

    aois_df = aois_df.split(by='line_idx')
    assert len(aois_df) == 2


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
            id='toy_text_1_1_aoi_sep',
        ),
    ],
)
def test_text_stimulus_splitting_unique_within(aoi_file, custom_read_kwargs):
    aois_df = text.from_file(
        aoi_file,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
        custom_read_kwargs=custom_read_kwargs,
    )

    aois_df = aois_df.split(by='line_idx')
    assert all(df.aois.n_unique(subset=['line_idx']) == 1 for df in aois_df)


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
            id='toy_text_1_1_aoi_sep',
        ),
    ],
)
def test_text_stimulus_splitting_different_between(aoi_file, custom_read_kwargs):
    aois_df = text.from_file(
        aoi_file,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
        custom_read_kwargs=custom_read_kwargs,
    )

    aois_df = aois_df.split(by='line_idx')
    unique_values = []
    for df in aois_df:
        unique_value = df.aois.unique(subset=['line_idx'])['line_idx'].to_list()
        unique_values.extend(unique_value)

    assert len(unique_values) == len(set(unique_values))


@pytest.fixture(name='text_stimulus')
def fixture_text_stimulus():
    yield text.from_file(
        'tests/files/toy_text_1_1_aoi.csv',
        aoi_column='word',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        end_x_column='bottom_left_x',
        end_y_column='bottom_left_y',
        page_column='page',
    )


@pytest.mark.parametrize(
    ('row', 'expected_aoi'),
    [
        pytest.param(
            {'x': 400, 'y': 125},
            'A',
            id='400,125',
        ),
        pytest.param(
            {'x': 500, 'y': 300},
            None,
            id='500,300',
        ),
    ],
)
def test_text_stimulus_get_aoi(text_stimulus, row, expected_aoi):
    aoi = text_stimulus.get_aoi(row=row, x_eye='x', y_eye='y')

    assert aoi['char'].first() == expected_aoi
