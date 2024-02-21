import pytest
import polars as pl
from polars.testing import assert_frame_equal
from pathlib import Path
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
