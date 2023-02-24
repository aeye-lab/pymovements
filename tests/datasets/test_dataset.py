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
"""Test all functionality in pymovements.datasets.dataset."""
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.datasets.dataset import Dataset
from pymovements.events.events import Event


def create_gaze_files_from_fileinfo(gaze_dfs, fileinfo, rootpath):
    rootpath.mkdir(parents=True, exist_ok=True)

    for gaze_df, fileinfo_row in zip(gaze_dfs, fileinfo.to_dicts()):
        filepath = fileinfo_row['filepath']

        for key in fileinfo_row.keys():
            if key in gaze_df.columns:
                gaze_df = gaze_df.drop(key)

        gaze_df.write_csv(rootpath / filepath)


def mock_toy(rootpath):
    subject_ids = list(range(1, 21))

    fileinfo = pl.DataFrame(data={'subject_id': subject_ids}, schema={'subject_id': pl.Int64})

    fileinfo = fileinfo.with_columns([
        pl.format('{}.csv', 'subject_id').alias('filepath'),
    ])

    fileinfo = fileinfo.sort(by='filepath')

    gaze_dfs = []
    for fileinfo_row in fileinfo.to_dicts():
        gaze_df = pl.from_dict(
            {
                'subject_id': fileinfo_row['subject_id'],
                'time': np.arange(1000),
            },
            schema={'subject_id': pl.Int64, 'time': pl.Int64},
        )
        gaze_dfs.append(gaze_df)

    create_gaze_files_from_fileinfo(gaze_dfs, fileinfo, rootpath / 'raw')

    return {
        'init_kwargs': {
            'root': rootpath,
            'filename_regex': r'(?P<subject_id>\d+).csv',
            'filename_regex_dtypes': {'subject_id': pl.Int64},
        },
        'fileinfo': fileinfo,
        'gaze_dfs': gaze_dfs,
    }


@pytest.fixture(name='dataset_configuration', params=['Toy'])
def fixture_dataset(request, tmp_path):
    rootpath = tmp_path

    if request.param == 'Toy':
        dataset_dict = mock_toy(rootpath)
    else:
        raise ValueError(f'{request.param} not supported as dataset mock')

    yield dataset_dict


def test_load_correct_fileinfo(dataset_configuration):
    dataset = Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()

    expected_fileinfo = dataset_configuration['fileinfo']
    assert_frame_equal(dataset.fileinfo, expected_fileinfo)


def test_load_correct_gaze_dfs(dataset_configuration):
    dataset = Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()

    expected_gaze_dfs = dataset_configuration['gaze_dfs']
    for result_gaze_df, expected_gaze_df in zip(dataset.gaze, expected_gaze_dfs):
        assert_frame_equal(result_gaze_df, expected_gaze_df)


@pytest.mark.parametrize(
    'init_kwargs, exception',
    [
        pytest.param(
            {'root': 'data', 'filename_regex': None},
            ValueError,
            id='filename_regex_none_value',
        ),
        pytest.param(
            {'root': 'data', 'filename_regex': 1},
            TypeError,
            id='filename_regex_wrong_type',
        ),
    ],
)
def test_init_exceptions(init_kwargs, exception):
    with pytest.raises(exception):
        Dataset(**init_kwargs)


@pytest.mark.parametrize(
    'init_kwargs, exception',
    [
        pytest.param(
            {'root': '/not/a/real/path'},
            RuntimeError,
            id='no_files_present',
        ),
    ],
)
def test_load_exceptions(init_kwargs, exception):
    dataset = Dataset(**init_kwargs)

    with pytest.raises(exception):
        dataset.load()


@pytest.mark.parametrize(
    'events_init, events_expected',
    [
        pytest.param(
            [],
            [],
            id='empty_list_stays_empty_list',
        ),
        pytest.param(
            [pl.DataFrame(schema=Event.schema)],
            [pl.DataFrame(schema=Event.schema)],
            id='empty_df_stays_empty_df',
        ),
        pytest.param(
            [pl.DataFrame({'type': ['event'], 'onset': [0], 'offset': [99]}, schema=Event.schema)],
            [pl.DataFrame(schema=Event.schema)],
            id='single_instance_filled_df_gets_cleared_to_empty_df',
        ),
        pytest.param(
            [
                pl.DataFrame(
                    {'type': ['event'], 'onset': [0], 'offset': [99]},
                    schema=Event.schema,
                ),
                pl.DataFrame(
                    {'type': ['event'], 'onset': [0], 'offset': [99]},
                    schema=Event.schema,
                ),
            ],
            [pl.DataFrame(schema=Event.schema), pl.DataFrame(schema=Event.schema)],
            id='two_instance_filled_df_gets_cleared_to_two_empty_dfs',
        ),
    ],
)
def test_clear_events(events_init, events_expected):
    dataset = Dataset(root='data')
    dataset.events = events_init
    dataset.clear_events()

    if isinstance(events_init, list) and not events_init:
        assert dataset.events == events_expected

    else:
        for events_df_result, events_df_expected in zip(dataset.events, events_expected):
            assert_frame_equal(events_df_result, events_df_expected)


@pytest.mark.parametrize(
    'init_kwargs, expected_paths',
    [
        pytest.param(
            {'root': '/data/set/path'},
            {
                'root': Path('/data/set/path/'),
                'path': Path('/data/set/path/'),
                'raw': Path('/data/set/path/raw'),
                'preprocessed': Path('/data/set/path/preprocessed'),
                'events': Path('/data/set/path/events'),
            },
        ),
        pytest.param(
            {'root': '/data/set/path', 'dataset_dirname': '.'},
            {
                'root': Path('/data/set/path/'),
                'path': Path('/data/set/path/'),
                'raw': Path('/data/set/path/raw'),
                'preprocessed': Path('/data/set/path/preprocessed'),
                'events': Path('/data/set/path/events'),
            },
        ),
        pytest.param(
            {'root': '/data/set/path', 'dataset_dirname': 'dataset'},
            {
                'root': Path('/data/set/path/'),
                'path': Path('/data/set/path/dataset'),
                'raw': Path('/data/set/path/dataset/raw'),
                'preprocessed': Path('/data/set/path/dataset/preprocessed'),
                'events': Path('/data/set/path/dataset/events'),
            },
        ),
        pytest.param(
            {
                'root': '/data/set/path', 'dataset_dirname': 'dataset',
                'events_dirname': 'custom_events',
            },
            {
                'root': Path('/data/set/path/'),
                'path': Path('/data/set/path/dataset'),
                'raw': Path('/data/set/path/dataset/raw'),
                'preprocessed': Path('/data/set/path/dataset/preprocessed'),
                'events': Path('/data/set/path/dataset/custom_events'),
            },
        ),
        pytest.param(
            {
                'root': '/data/set/path', 'dataset_dirname': 'dataset',
                'preprocessed_dirname': 'custom_preprocessed',
            },
            {
                'root': Path('/data/set/path/'),
                'path': Path('/data/set/path/dataset'),
                'raw': Path('/data/set/path/dataset/raw'),
                'preprocessed': Path('/data/set/path/dataset/custom_preprocessed'),
                'events': Path('/data/set/path/dataset/events'),
            },
        ),
        pytest.param(
            {
                'root': '/data/set/path', 'dataset_dirname': 'dataset',
                'raw_dirname': 'custom_raw',
            },
            {
                'root': Path('/data/set/path/'),
                'path': Path('/data/set/path/dataset'),
                'raw': Path('/data/set/path/dataset/custom_raw'),
                'preprocessed': Path('/data/set/path/dataset/preprocessed'),
                'events': Path('/data/set/path/dataset/events'),
            },
        ),
    ],
)
def test_paths(init_kwargs, expected_paths):
    dataset = Dataset(**init_kwargs)

    assert dataset.root == expected_paths['root']
    assert dataset.path == expected_paths['path']
    assert dataset.raw_rootpath == expected_paths['raw']
    assert dataset.preprocessed_rootpath == expected_paths['preprocessed']
    assert dataset.events_rootpath == expected_paths['events']
