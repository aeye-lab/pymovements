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

from pathlib import Path

import pytest

import pymovements as pm


@pytest.mark.parametrize(
    ('aoi_file'),
    [
        pytest.param(
            'tests/files/toy_text_1_1_aoi.csv',
            id='toy_text_1_1_aoi',
        ),
        pytest.param(
            'tests/files/toy_text_2_2_aoi.csv',
            id='toy_text_2_2_aoi',
        ),
        pytest.param(
            'tests/files/toy_text_3_3_aoi.csv',
            id='toy_text_3_3_aoi',
        ),
    ],
)
def test_str_aoi_path(aoi_file):
    pm.stimulus.text.from_file(
        aoi_file,
        character_column='char',
        pixel_x_column='top_left_x',
        pixel_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
    )


@pytest.mark.parametrize(
    ('aoi_file'),
    [
        pytest.param(
            'tests/files/toy_text_1_1_aoi.csv',
            id='toy_text_1_1_aoi',
        ),
        pytest.param(
            'tests/files/toy_text_2_2_aoi.csv',
            id='toy_text_2_2_aoi',
        ),
        pytest.param(
            'tests/files/toy_text_3_3_aoi.csv',
            id='toy_text_3_3_aoi',
        ),
    ],
)
def test_str_aoi_path_kwargs(aoi_file):
    pm.stimulus.text.from_file(
        aoi_file,
        character_column='char',
        pixel_x_column='top_left_x',
        pixel_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
        custom_read_kwargs={
            'separator': ',',
        },
    )


@pytest.mark.parametrize(
    ('aoi_file'),
    [
        pytest.param(
            Path('tests/files/toy_text_1_1_aoi.csv'),
            id='toy_text_1_1_aoi',
        ),
        pytest.param(
            Path('tests/files/toy_text_2_2_aoi.csv'),
            id='toy_text_2_2_aoi',
        ),
        pytest.param(
            Path('tests/files/toy_text_3_3_aoi.csv'),
            id='toy_text_3_3_aoi',
        ),
    ],
)
def test_Path_aoi_path(aoi_file):
    pm.stimulus.text.from_file(
        aoi_file,
        character_column='char',
        pixel_x_column='top_left_x',
        pixel_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
    )


@pytest.mark.parametrize(
    ('aoi_file'),
    [
        pytest.param(
            Path('tests/files/toy_text_1_1_aoi.csv'),
            id='toy_text_1_1_aoi',
        ),
        pytest.param(
            Path('tests/files/toy_text_2_2_aoi.csv'),
            id='toy_text_2_2_aoi',
        ),
        pytest.param(
            Path('tests/files/toy_text_3_3_aoi.csv'),
            id='toy_text_3_3_aoi',
        ),
    ],
)
def test_Path_aoi_path_kwargs(aoi_file):
    pm.stimulus.text.from_file(
        aoi_file,
        character_column='char',
        pixel_x_column='top_left_x',
        pixel_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
        custom_read_kwargs={
            'separator': ',',
        },
    )


def test_text_stimulus_unsupported_format():
    with pytest.raises(ValueError) as excinfo:
        pm.stimulus.text.from_file(
            'tests/files/toy_text_1_1_aoi.pickle',
            character_column='char',
            pixel_x_column='top_left_x',
            pixel_y_column='top_left_y',
            width_column='width',
            height_column='height',
            page_column='page',
        )
    msg, = excinfo.value.args
    assert msg == 'unsupported file format ".pickle".Supported formats are: [\'.csv\', \'.tsv\', \'.txt\']'
