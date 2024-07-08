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
"""Test all GazeDataFrame functionality."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm

EXPECTED_DF = {
    'char_left_pixel': pl.DataFrame(
        [
            (1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, 'e'),
            (1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, 'e'),
            (1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, 'e'),
            (1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, 'e'),
            (1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, 'e'),
            (1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, 'e'),
            (1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, 'e'),
            (1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, 'e'),
            (1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, ''),
            (1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, 'e'),
        ],
        schema=[
            'trialId',
            'pointId',
            'time',
            'pixel_xl',
            'pixel_yl',
            'pixel_xr',
            'pixel_yr',
            'area_of_interest',
        ],
    ),
    'char_right_pixel': pl.DataFrame(
        [
            (1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, 'e'),
            (1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, 'e'),
            (1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, 'e'),
            (1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, 'e'),
            (1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, 'e'),
            (1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, 'e'),
            (1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, 'e'),
            (1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, 'e'),
            (1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, 'e'),
            (1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, 'e'),
        ],
        schema=[
            'trialId',
            'pointId',
            'time',
            'pixel_xl',
            'pixel_yl',
            'pixel_xr',
            'pixel_yr',
            'area_of_interest',
        ],
    ),

    'word_left_pixel': pl.DataFrame(
        [
            (1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, 'files,'),
            (1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, 'files,'),
            (1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, 'files,'),
            (1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, 'files,'),
            (1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, 'files,'),
            (1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, 'files,'),
            (1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, 'files,'),
            (1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, 'files,'),
            (1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, ''),
            (1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, 'files,'),
        ],
        schema=[
            'trialId',
            'pointId',
            'time',
            'pixel_xl',
            'pixel_yl',
            'pixel_xr',
            'pixel_yr',
            'area_of_interest',
        ],
    ),
    'word_right_pixel': pl.DataFrame(
        [
            (1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, 'files,'),
            (1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, 'files,'),
            (1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, 'files,'),
            (1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, 'files,'),
            (1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, 'files,'),
            (1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, 'files,'),
            (1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, 'files,'),
            (1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, 'files,'),
            (1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, 'files,'),
            (1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, 'files,'),
        ],
        schema=[
            'trialId',
            'pointId',
            'time',
            'pixel_xl',
            'pixel_yl',
            'pixel_xr',
            'pixel_yr',
            'area_of_interest',
        ],
    ),
    'char_left_position': pl.DataFrame(
        [
            (1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, 'e'),
            (1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, 'e'),
            (1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, 'e'),
            (1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, 'e'),
            (1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, 'e'),
            (1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, 'e'),
            (1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, 'e'),
            (1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, 'e'),
            (1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, ''),
            (1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, 'e'),
        ],
        schema=[
            'trialId',
            'pointId',
            'time',
            'position_xl',
            'position_yl',
            'position_xr',
            'position_yr',
            'area_of_interest',
        ],
    ),
    'char_right_position': pl.DataFrame(
        [
            (1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, 'e'),
            (1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, 'e'),
            (1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, 'e'),
            (1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, 'e'),
            (1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, 'e'),
            (1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, 'e'),
            (1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, 'e'),
            (1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, 'e'),
            (1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, 'e'),
            (1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, 'e'),
        ],
        schema=[
            'trialId',
            'pointId',
            'time',
            'position_xl',
            'position_yl',
            'position_xr',
            'position_yr',
            'area_of_interest',
        ],
    ),

    'word_left_position': pl.DataFrame(
        [
            (1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, 'files,'),
            (1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, 'files,'),
            (1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, 'files,'),
            (1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, 'files,'),
            (1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, 'files,'),
            (1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, 'files,'),
            (1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, 'files,'),
            (1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, 'files,'),
            (1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, ''),
            (1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, 'files,'),
        ],
        schema=[
            'trialId',
            'pointId',
            'time',
            'position_xl',
            'position_yl',
            'position_xr',
            'position_yr',
            'area_of_interest',
        ],
    ),
    'word_right_position': pl.DataFrame(
        [
            (1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, 'files,'),
            (1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, 'files,'),
            (1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, 'files,'),
            (1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, 'files,'),
            (1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, 'files,'),
            (1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, 'files,'),
            (1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, 'files,'),
            (1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, 'files,'),
            (1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, 'files,'),
            (1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, 'files,'),
        ],
        schema=[
            'trialId',
            'pointId',
            'time',
            'position_xl',
            'position_yl',
            'position_xr',
            'position_yr',
            'area_of_interest',
        ],
    ),
    'char_auto_pixel': pl.DataFrame(
        [
            (1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, 'e'),
            (1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, 'e'),
            (1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, 'e'),
            (1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, 'e'),
            (1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, 'e'),
            (1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, 'e'),
            (1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, 'e'),
            (1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, 'e'),
            (1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, 'e'),
            (1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, 'e'),
        ],
        schema=[
            'trialId',
            'pointId',
            'time',
            'pixel_xl',
            'pixel_yl',
            'pixel_xr',
            'pixel_yr',
            'area_of_interest',
        ],
    ),
    'char_auto_position': pl.DataFrame(
        [
            (1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, 'e'),
            (1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, 'e'),
            (1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, 'e'),
            (1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, 'e'),
            (1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, 'e'),
            (1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, 'e'),
            (1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, 'e'),
            (1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, 'e'),
            (1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, 'e'),
            (1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, 'e'),
        ],
        schema=[
            'trialId',
            'pointId',
            'time',
            'position_xl',
            'position_yl',
            'position_xr',
            'position_yr',
            'area_of_interest',
        ],
    ),

    'word_auto_pixel': pl.DataFrame(
        [
            (1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, 'files,'),
            (1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, 'files,'),
            (1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, 'files,'),
            (1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, 'files,'),
            (1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, 'files,'),
            (1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, 'files,'),
            (1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, 'files,'),
            (1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, 'files,'),
            (1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, 'files,'),
            (1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, 'files,'),
        ],
        schema=[
            'trialId',
            'pointId',
            'time',
            'pixel_xl',
            'pixel_yl',
            'pixel_xr',
            'pixel_yr',
            'area_of_interest',
        ],
    ),
    'word_auto_position': pl.DataFrame(
        [
            (1, 1, 8005274, 649.5, 531.1, 640.6, 529.1, 'files,'),
            (1, 1, 8005275, 649.8, 533.2, 639.7, 528.9, 'files,'),
            (1, 1, 8005276, 647.7, 534.0, 640.6, 529.3, 'files,'),
            (1, 1, 8005277, 646.2, 533.0, 642.1, 531.3, 'files,'),
            (1, 1, 8005278, 646.5, 533.7, 642.9, 531.0, 'files,'),
            (1, 1, 8005279, 647.2, 534.6, 642.6, 531.6, 'files,'),
            (1, 1, 8005280, 647.3, 534.0, 642.3, 530.6, 'files,'),
            (1, 1, 8005281, 647.7, 536.3, 642.2, 529.4, 'files,'),
            (1, 1, 8005282, 647.5, 537.0, 641.4, 531.3, 'files,'),
            (1, 1, 8005283, 648.3, 534.9, 640.9, 529.0, 'files,'),
        ],
        schema=[
            'trialId',
            'pointId',
            'time',
            'position_xl',
            'position_yl',
            'position_xr',
            'position_yr',
            'area_of_interest',
        ],
    ),
}


@pytest.mark.parametrize(
    ('eye'),
    [
        'right',
        'left',
        'auto',
    ],
)
@pytest.mark.parametrize(
    ('aoi_column'),
    [
        'word',
        'char',
    ],
)
@pytest.mark.parametrize(
    ('column_type'),
    [
        'pixel',
        'position',
    ],
)
def test_gaze_to_aoi_mapping_char(eye, aoi_column, column_type):
    aoi_df = pm.stimulus.text.from_file(
        'tests/files/toy_text_1_1_aoi.csv',
        aoi_column=aoi_column,
        pixel_x_column='top_left_x',
        pixel_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
    )
    if column_type == 'pixel':
        gaze_df = pm.gaze.io.from_csv(
            'tests/files/judo1000_example.csv',
            **{'separator': '\t'},
            pixel_columns=['x_left', 'y_left', 'x_right', 'y_right'],
        )
    elif column_type == 'position':
        gaze_df = pm.gaze.io.from_csv(
            'tests/files/judo1000_example.csv',
            **{'separator': '\t'},
            position_columns=['x_left', 'y_left', 'x_right', 'y_right'],
        )

    gaze_df.map_to_aois(aoi_df, eye=eye)
    assert_frame_equal(gaze_df.frame, EXPECTED_DF[f'{aoi_column}_{eye}_{column_type}'])


def test_map_to_aois_raises_value_error():
    aoi_df = pm.stimulus.text.from_file(
        'tests/files/toy_text_1_1_aoi.csv',
        aoi_column='char',
        pixel_x_column='top_left_x',
        pixel_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
    )
    gaze_df = pm.gaze.io.from_csv(
        'tests/files/judo1000_example.csv',
        **{'separator': '\t'},
    )

    with pytest.raises(ValueError) as excinfo:
        gaze_df.map_to_aois(aoi_df, eye='right')
    msg, = excinfo.value.args
    assert msg == 'neither position nor pixel in gaze dataframe, one needed for mapping'
