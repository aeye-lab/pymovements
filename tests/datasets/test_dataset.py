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
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pymovements.base import Experiment
from pymovements.datasets.dataset import Dataset
from pymovements.events.engbert import microsaccades
from pymovements.events.events import Event
from pymovements.events.events import Fixation
from pymovements.events.events import Saccade
from pymovements.events.ivt import ivt


def create_raw_gaze_files_from_fileinfo(gaze_dfs, fileinfo, rootpath):
    rootpath.mkdir(parents=True, exist_ok=True)

    for gaze_df, fileinfo_row in zip(gaze_dfs, fileinfo.to_dicts()):
        filepath = fileinfo_row['filepath']

        for key in fileinfo_row.keys():
            if key in gaze_df.columns:
                gaze_df = gaze_df.drop(key)

        gaze_df.write_csv(rootpath / filepath)


def create_preprocessed_gaze_files_from_fileinfo(gaze_dfs, fileinfo, rootpath):
    rootpath.mkdir(parents=True, exist_ok=True)

    for gaze_df, fileinfo_row in zip(gaze_dfs, fileinfo.to_dicts()):
        filepath = fileinfo_row['filepath']
        filepath = filepath.replace('csv', 'feather')

        for key in fileinfo_row.keys():
            if key in gaze_df.columns:
                gaze_df = gaze_df.drop(key)

        gaze_df.write_ipc(rootpath / filepath)


def create_event_files_from_fileinfo(gaze_dfs, fileinfo, rootpath):
    rootpath.mkdir(parents=True, exist_ok=True)

    for gaze_df, fileinfo_row in zip(gaze_dfs, fileinfo.to_dicts()):
        filepath = fileinfo_row['filepath']
        filepath = filepath.replace('csv', 'feather')

        for key in fileinfo_row.keys():
            if key in gaze_df.columns:
                gaze_df = gaze_df.drop(key)

        gaze_df.write_ipc(rootpath / filepath)


def mock_toy(rootpath, raw_fileformat='csv'):
    subject_ids = list(range(1, 21))

    fileinfo = pl.DataFrame(data={'subject_id': subject_ids}, schema={'subject_id': pl.Int64})

    fileinfo = fileinfo.with_columns([
        pl.format('{}.' + raw_fileformat, 'subject_id').alias('filepath'),
    ])

    fileinfo = fileinfo.sort(by='filepath')

    gaze_dfs = []
    for fileinfo_row in fileinfo.to_dicts():
        gaze_df = pl.from_dict(
            {
                'subject_id': fileinfo_row['subject_id'],
                'time': np.arange(1000),
                'x_left_pix': np.zeros(1000),
                'y_left_pix': np.zeros(1000),
                'x_right_pix': np.zeros(1000),
                'y_right_pix': np.zeros(1000),
            },
            schema={
                'subject_id': pl.Int64,
                'time': pl.Int64,
                'x_left_pix': pl.Float64,
                'y_left_pix': pl.Float64,
                'x_right_pix': pl.Float64,
                'y_right_pix': pl.Float64,
            },
        )
        gaze_dfs.append(gaze_df)

    create_raw_gaze_files_from_fileinfo(gaze_dfs, fileinfo, rootpath / 'raw')

    preprocessed_gaze_dfs = []
    for fileinfo_row in fileinfo.to_dicts():
        gaze_df = pl.from_dict(
            {
                'subject_id': fileinfo_row['subject_id'],
                'time': np.arange(1000),
                'x_left_pix': np.zeros(1000),
                'y_left_pix': np.zeros(1000),
                'x_right_pix': np.zeros(1000),
                'y_right_pix': np.zeros(1000),
                'x_left_dva': np.zeros(1000),
                'y_left_dva': np.zeros(1000),
                'x_right_dva': np.zeros(1000),
                'y_right_dva': np.zeros(1000),
                'x_left_vel': np.zeros(1000),
                'y_left_vel': np.zeros(1000),
                'x_right_vel': np.zeros(1000),
                'y_right_vel': np.zeros(1000),
            },
            schema={
                'subject_id': pl.Int64,
                'time': pl.Int64,
                'x_left_pix': pl.Float64,
                'y_left_pix': pl.Float64,
                'x_right_pix': pl.Float64,
                'y_right_pix': pl.Float64,
                'x_left_dva': pl.Float64,
                'y_left_dva': pl.Float64,
                'x_right_dva': pl.Float64,
                'y_right_dva': pl.Float64,
                'x_left_vel': pl.Float64,
                'y_left_vel': pl.Float64,
                'x_right_vel': pl.Float64,
                'y_right_vel': pl.Float64,
            },
        )
        preprocessed_gaze_dfs.append(gaze_df)

    create_preprocessed_gaze_files_from_fileinfo(
        preprocessed_gaze_dfs, fileinfo, rootpath / 'preprocessed',
    )

    event_dfs = []
    for fileinfo_row in fileinfo.to_dicts():
        event_df = pl.from_dict(
            {
                'subject_id': fileinfo_row['subject_id'],
                'type': 'saccade',
                'onset': np.arange(0, 901, 100),
                'offset': np.arange(10, 911, 100),
            },
            schema={'subject_id': pl.Int64, 'type': pl.Utf8, 'onset': pl.Int64, 'offset': pl.Int64},
        )
        event_dfs.append(event_df)

    create_event_files_from_fileinfo(event_dfs, fileinfo, rootpath / 'events')

    return {
        'init_kwargs': {
            'root': rootpath,
            'experiment': Experiment(
                screen_width_px=1280,
                screen_height_px=1024,
                screen_width_cm=38,
                screen_height_cm=30.2,
                distance_cm=68,
                origin='lower left',
                sampling_rate=1000,
            ),
            'filename_regex': r'(?P<subject_id>\d+).' + raw_fileformat,
            'filename_regex_dtypes': {'subject_id': pl.Int64},
        },
        'fileinfo': fileinfo,
        'raw_gaze_dfs': gaze_dfs,
        'preprocessed_gaze_dfs': preprocessed_gaze_dfs,
        'event_dfs': event_dfs,
    }


@pytest.fixture(name='dataset_configuration', params=['Toy'])
def fixture_dataset(request, tmp_path):
    rootpath = tmp_path

    if request.param == 'Toy':
        dataset_dict = mock_toy(rootpath, raw_fileformat='csv')
    elif request.param == 'ToyMat':
        dataset_dict = mock_toy(rootpath, raw_fileformat='mat')
    else:
        raise ValueError(f'{request.param} not supported as dataset mock')

    yield dataset_dict


def test_load_correct_fileinfo(dataset_configuration):
    dataset = Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()

    expected_fileinfo = dataset_configuration['fileinfo']
    assert_frame_equal(dataset.fileinfo, expected_fileinfo)


def test_load_correct_raw_gaze_dfs(dataset_configuration):
    dataset = Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()

    expected_gaze_dfs = dataset_configuration['raw_gaze_dfs']
    for result_gaze_df, expected_gaze_df in zip(dataset.gaze, expected_gaze_dfs):
        assert_frame_equal(result_gaze_df, expected_gaze_df)


def test_load_correct_preprocessed_gaze_dfs(dataset_configuration):
    dataset = Dataset(**dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True)

    expected_gaze_dfs = dataset_configuration['preprocessed_gaze_dfs']
    for result_gaze_df, expected_gaze_df in zip(dataset.gaze, expected_gaze_dfs):
        assert_frame_equal(result_gaze_df, expected_gaze_df)


def test_load_correct_event_dfs(dataset_configuration):
    dataset = Dataset(**dataset_configuration['init_kwargs'])
    dataset.load(events=True)

    expected_event_dfs = dataset_configuration['event_dfs']
    for result_event_df, expected_event_df in zip(dataset.events, expected_event_dfs):
        assert_frame_equal(result_event_df, expected_event_df)


@pytest.mark.parametrize(
    'subset, fileinfo_idx',
    [
        pytest.param(
            {'subject_id': 1},
            [0],
            id='subset_key_not_in_fileinfo',
        ),
        pytest.param(
            {'subject_id': [1, 11, 12]},
            [0, 2, 3],
            id='subset_key_not_in_fileinfo',
        ),
    ],
)
def test_load_subset(subset, fileinfo_idx, dataset_configuration):
    dataset = Dataset(**dataset_configuration['init_kwargs'])
    dataset.load(subset=subset)

    expected_fileinfo = dataset_configuration['fileinfo']
    expected_fileinfo = expected_fileinfo[fileinfo_idx]

    assert_frame_equal(dataset.fileinfo, expected_fileinfo)


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
    'init_kwargs, load_kwargs, exception',
    [
        pytest.param(
            {'root': '/not/a/real/path'},
            {},
            RuntimeError,
            id='no_files_present',
        ),
        pytest.param(
            {},
            {'subset': 1},
            TypeError,
            id='subset_no_dict',
        ),
        pytest.param(
            {},
            {'subset': {1: 1}},
            TypeError,
            id='subset_no_str_key',
        ),
        pytest.param(
            {},
            {'subset': {'unknown': 1}},
            ValueError,
            id='subset_key_not_in_fileinfo',
        ),
        pytest.param(
            {},
            {'subset': {'subject_id': None}},
            TypeError,
            id='subset_value_invalid_type',
        ),
    ],
)
def test_load_exceptions(init_kwargs, load_kwargs, exception, dataset_configuration):
    init_kwargs = {**dataset_configuration['init_kwargs'], **init_kwargs}
    dataset = Dataset(**init_kwargs)

    with pytest.raises(exception):
        dataset.load(**load_kwargs)


@pytest.mark.parametrize(
    'exception',
    [
        pytest.param(RuntimeError, id='matlab_dataset'),
    ],
)
@pytest.mark.parametrize('dataset_configuration', ['ToyMat'], indirect=['dataset_configuration'])
def test_load_mat_file_exception(exception, dataset_configuration):
    dataset = Dataset(**dataset_configuration['init_kwargs'])

    with pytest.raises(exception):
        dataset.load()


def test_pix2deg(dataset_configuration):
    dataset = Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()

    original_schema = dataset.gaze[0].schema

    dataset.pix2deg()

    expected_schema = {
        **original_schema,
        'x_left_dva': pl.Float64,
        'y_left_dva': pl.Float64,
        'x_right_dva': pl.Float64,
        'y_right_dva': pl.Float64,
    }

    for result_gaze_df in dataset.gaze:
        assert result_gaze_df.schema == expected_schema


def test_pos2vel(dataset_configuration):
    dataset = Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()
    dataset.pix2deg()

    original_schema = dataset.gaze[0].schema

    dataset.pos2vel()

    expected_schema = {
        **original_schema,
        'x_left_vel': pl.Float64,
        'y_left_vel': pl.Float64,
        'x_right_vel': pl.Float64,
        'y_right_vel': pl.Float64,
    }

    for result_gaze_df in dataset.gaze:
        assert result_gaze_df.schema == expected_schema


@pytest.mark.parametrize(
    'detect_event_kwargs',
    [
        pytest.param(
            {
                'method': microsaccades,
                'threshold': 1,
                'eye': 'auto',
            },
            id='1',
        ),
        pytest.param(
            {
                'method': microsaccades,
                'threshold': 1,
                'eye': 'left',
            },
            id='2',
        ),
        pytest.param(
            {
                'method': microsaccades,
                'threshold': 1,
                'eye': 'right',
            },
            id='3',
        ),
    ],
)
def test_detect_events(detect_event_kwargs, dataset_configuration):
    dataset = Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()
    dataset.pix2deg()
    dataset.pos2vel()
    dataset.detect_events(**detect_event_kwargs)

    expected_schema = {'subject_id': pl.Int64, **Saccade.schema}
    for result_event_df in dataset.events:
        assert result_event_df.schema == expected_schema


@pytest.mark.parametrize(
    'detect_event_kwargs_1, detect_event_kwargs_2, expected_schema',
    [
        pytest.param(
            {
                'method': microsaccades,
                'threshold': 1,
                'eye': 'auto',
            },
            {
                'method': microsaccades,
                'threshold': 1,
                'eye': 'auto',
            },
            {'subject_id': pl.Int64, **Saccade.schema},
            id='two-saccade-runs',
        ),
        pytest.param(
            {
                'method': microsaccades,
                'threshold': 1,
                'eye': 'left',
            },
            {
                'method': ivt,
                'velocity_threshold': 1,
                'minimum_duration': 1,
            },
            {'subject_id': pl.Int64, **Saccade.schema, **Fixation.schema},
            id='one-saccade-one-fixation-run',
        ),
    ],
)
def test_detect_events_multiple_calls(
    detect_event_kwargs_1, detect_event_kwargs_2,
    expected_schema, dataset_configuration,
):
    dataset = Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()
    dataset.pix2deg()
    dataset.pos2vel()
    dataset.detect_events(**detect_event_kwargs_1)
    dataset.detect_events(**detect_event_kwargs_2)

    for result_event_df in dataset.events:
        assert result_event_df.schema == expected_schema


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
    'detect_event_kwargs, events_dirname, expected_save_dirpath',
    [
        pytest.param(
            {'method': microsaccades, 'threshold': 1, 'eye': 'auto'},
            None,
            'events',
            id='none_dirname',
        ),
        pytest.param(
            {'method': microsaccades, 'threshold': 1, 'eye': 'auto'},
            'events_test',
            'events_test',
            id='explicit_dirname',
        ),
    ],
)
def test_save_events(
    detect_event_kwargs, events_dirname, expected_save_dirpath,
    dataset_configuration,
):
    dataset = Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()
    dataset.pix2deg()
    dataset.pos2vel()
    dataset.detect_events(**detect_event_kwargs)

    if events_dirname is None:
        events_dirname = 'events'
    shutil.rmtree(dataset.path / Path(events_dirname), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(expected_save_dirpath), ignore_errors=True)
    dataset.save_events(events_dirname)

    assert (dataset.path / expected_save_dirpath).is_dir(), (
        f'data was not written to {dataset.path / Path(expected_save_dirpath)}'
    )


@pytest.mark.parametrize(
    'preprocessed_dirname, expected_save_dirpath',
    [
        pytest.param(
            None,
            'preprocessed',
            id='none_dirname',
        ),
        pytest.param(
            'preprocessed_test',
            'preprocessed_test',
            id='explicit_dirname',
        ),
    ],
)
def test_save_preprocessed(preprocessed_dirname, expected_save_dirpath, dataset_configuration):
    dataset = Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()
    dataset.pix2deg()
    dataset.pos2vel()

    if preprocessed_dirname is None:
        preprocessed_dirname = 'preprocessed'
    shutil.rmtree(dataset.path / Path(preprocessed_dirname), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(expected_save_dirpath), ignore_errors=True)
    dataset.save_preprocessed(preprocessed_dirname)

    assert (dataset.path / expected_save_dirpath).is_dir(), (
        f'data was not written to {dataset.path / Path(expected_save_dirpath)}'
    )


@pytest.mark.parametrize(
    'expected_save_preprocessed_path, expected_save_events_path, save_kwargs',
    [
        pytest.param(
            'preprocessed',
            'events',
            {},
            id='none_dirname',
        ),
        pytest.param(
            'preprocessed_test',
            'events',
            {'preprocessed_dirname': 'preprocessed_test'},
            id='none_dirname',
        ),
        pytest.param(
            'preprocessed',
            'events_test',
            {'events_dirname': 'events_test'},
            id='none_dirname',
        ),
    ],
)
def test_save(
    expected_save_preprocessed_path, expected_save_events_path,
    save_kwargs, dataset_configuration,
):
    dataset = Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()
    dataset.pix2deg()
    dataset.pos2vel()

    detect_events_kwargs = {'method': microsaccades, 'threshold': 1, 'eye': 'auto'}
    dataset.detect_events(**detect_events_kwargs)

    preprocessed_dirname = save_kwargs.get('preprocessed_dirname', 'preprocessed')
    events_dirname = save_kwargs.get('events_dirname', 'events')

    shutil.rmtree(dataset.path / Path(preprocessed_dirname), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(expected_save_preprocessed_path), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(events_dirname), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(expected_save_events_path), ignore_errors=True)
    dataset.save(preprocessed_dirname, events_dirname)

    assert (dataset.path / Path(expected_save_preprocessed_path)).is_dir(), (
        f'data was not written to {dataset.path / Path(expected_save_events_path)}'
    )
    assert (dataset.path / Path(expected_save_events_path)).is_dir(), (
        f'data was not written to {dataset.path / Path(expected_save_events_path)}'
    )


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


@pytest.mark.parametrize(
    'new_fileinfo, exception',
    [
        pytest.param(None, AttributeError),
        pytest.param([], AttributeError),
    ],
)
def test_check_fileinfo(new_fileinfo, exception):
    dataset = Dataset('data')

    dataset.fileinfo = new_fileinfo

    with pytest.raises(exception):
        dataset._check_fileinfo()


@pytest.mark.parametrize(
    'new_gaze, exception',
    [
        pytest.param(None, AttributeError),
        pytest.param([], AttributeError),
    ],
)
def test_check_gaze_dataframe(new_gaze, exception):
    dataset = Dataset('data')

    dataset.gaze = new_gaze

    with pytest.raises(exception):
        dataset._check_gaze_dataframe()


def test_check_experiment():
    dataset = Dataset('data')
    with pytest.raises(AttributeError):
        dataset._check_experiment()
