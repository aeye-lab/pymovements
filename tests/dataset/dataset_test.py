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
"""Test all functionality in pymovements.dataset.dataset."""
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm


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


def mock_toy(rootpath, raw_fileformat, eyes):
    subject_ids = list(range(1, 21))

    fileinfo = pl.DataFrame(data={'subject_id': subject_ids}, schema={'subject_id': pl.Int64})

    fileinfo = fileinfo.with_columns([
        pl.format('{}.' + raw_fileformat, 'subject_id').alias('filepath'),
    ])

    fileinfo = fileinfo.sort(by='filepath')

    gaze_dfs = []
    for fileinfo_row in fileinfo.to_dicts():
        if eyes == 'both':
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
            pixel_columns = ['x_left_pix', 'y_left_pix', 'x_right_pix', 'y_right_pix']

        elif eyes == 'left':
            gaze_df = pl.from_dict(
                {
                    'subject_id': fileinfo_row['subject_id'],
                    'time': np.arange(1000),
                    'x_left_pix': np.zeros(1000),
                    'y_left_pix': np.zeros(1000),
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'x_left_pix': pl.Float64,
                    'y_left_pix': pl.Float64,
                },
            )
            pixel_columns = ['x_left_pix', 'y_left_pix']
        elif eyes == 'right':
            gaze_df = pl.from_dict(
                {
                    'subject_id': fileinfo_row['subject_id'],
                    'time': np.arange(1000),
                    'x_right_pix': np.zeros(1000),
                    'y_right_pix': np.zeros(1000),
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'x_right_pix': pl.Float64,
                    'y_right_pix': pl.Float64,
                },
            )
            pixel_columns = ['x_right_pix', 'y_right_pix']
        elif eyes == 'none':
            gaze_df = pl.from_dict(
                {
                    'subject_id': fileinfo_row['subject_id'],
                    'time': np.arange(1000),
                    'x_pix': np.zeros(1000),
                    'y_pix': np.zeros(1000),
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'x_pix': pl.Float64,
                    'y_pix': pl.Float64,
                },
            )
            pixel_columns = ['x_pix', 'y_pix']
        else:
            raise ValueError(f'invalid value for eyes: {eyes}')

        gaze_dfs.append(gaze_df)

    create_raw_gaze_files_from_fileinfo(gaze_dfs, fileinfo, rootpath / 'raw')

    # Create GazeDataFrames for passing as ground truth
    gaze_dfs = [
        pm.GazeDataFrame(gaze_df, pixel_columns=pixel_columns)
        for gaze_df in gaze_dfs
    ]

    preprocessed_gaze_dfs = []
    for fileinfo_row in fileinfo.to_dicts():
        position_columns = [pixel_column.replace('pix', 'pos') for pixel_column in pixel_columns]
        velocity_columns = [pixel_column.replace('pix', 'vel') for pixel_column in pixel_columns]
        acceleration_columns = [
            pixel_column.replace('pix', 'acc') for pixel_column in pixel_columns
        ]

        gaze_data = {
            'subject_id': fileinfo_row['subject_id'],
            'time': np.arange(1000),
        }
        gaze_schema = {
            'subject_id': pl.Int64,
            'time': pl.Int64,
        }

        for column in pixel_columns + position_columns + velocity_columns + acceleration_columns:
            gaze_data[column] = np.zeros(1000)
            gaze_schema[column] = pl.Float64

        gaze_df = pl.from_dict(gaze_data, schema=gaze_schema)
        preprocessed_gaze_dfs.append(gaze_df)

    create_preprocessed_gaze_files_from_fileinfo(
        preprocessed_gaze_dfs, fileinfo, rootpath / 'preprocessed',
    )

    # Create GazeDataFrames for passing as ground truth
    preprocessed_gaze_dfs = [
        pm.GazeDataFrame(
            preprocessed_gaze_df,
            pixel_columns=pixel_columns,
            # position_columns=position_columns,
            # velocity_columns=velocity_columns,
            # acceleration_columns=acceleration_columns,
        )
        for preprocessed_gaze_df in preprocessed_gaze_dfs
    ]

    event_dfs = []
    for fileinfo_row in fileinfo.to_dicts():
        event_df = pl.from_dict(
            {
                'subject_id': fileinfo_row['subject_id'],
                'name': ['saccade', 'fixation'] * 5,
                'onset': np.arange(0, 100, 10),
                'offset': np.arange(5, 105, 10),
                'duration': np.array([5] * 10),
            },
            schema={
                'subject_id': pl.Int64,
                'name': pl.Utf8,
                'onset': pl.Int64,
                'offset': pl.Int64,
                'duration': pl.Int64,
            },
        )
        event_dfs.append(event_df)

    create_event_files_from_fileinfo(event_dfs, fileinfo, rootpath / 'events')

    dataset_definition = pm.DatasetDefinition(
        experiment=pm.Experiment(
            screen_width_px=1280,
            screen_height_px=1024,
            screen_width_cm=38,
            screen_height_cm=30.2,
            distance_cm=68,
            origin='lower left',
            sampling_rate=1000,
        ),
        filename_format=r'{subject_id:d}.' + raw_fileformat,
        filename_format_dtypes={'subject_id': pl.Int64},
        time_column='time',
        pixel_columns=pixel_columns,
    )

    return {
        'init_kwargs': {
            'definition': dataset_definition,
            'path': pm.DatasetPaths(root=rootpath, dataset='.'),
        },
        'fileinfo': fileinfo,
        'raw_gaze_dfs': gaze_dfs,
        'preprocessed_gaze_dfs': preprocessed_gaze_dfs,
        'event_dfs': event_dfs,
        'eyes': eyes,
    }


@pytest.fixture(name='dataset_configuration', params=['ToyMono', 'ToyBino', 'ToyLeft', 'ToyRight'])
def fixture_dataset(request, tmp_path):
    rootpath = tmp_path

    if request.param == 'ToyBino':
        dataset_dict = mock_toy(rootpath, raw_fileformat='csv', eyes='both')
    elif request.param == 'ToyMono':
        dataset_dict = mock_toy(rootpath, raw_fileformat='csv', eyes='none')
    elif request.param == 'ToyLeft':
        dataset_dict = mock_toy(rootpath, raw_fileformat='csv', eyes='left')
    elif request.param == 'ToyRight':
        dataset_dict = mock_toy(rootpath, raw_fileformat='csv', eyes='right')
    elif request.param == 'ToyMat':
        dataset_dict = mock_toy(rootpath, raw_fileformat='mat', eyes='both')
    else:
        raise ValueError(f'{request.param} not supported as dataset mock')

    yield dataset_dict


def test_load_correct_fileinfo(dataset_configuration):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()

    expected_fileinfo = dataset_configuration['fileinfo']
    assert_frame_equal(dataset.fileinfo, expected_fileinfo)


def test_load_correct_raw_gaze_dfs(dataset_configuration):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()

    expected_gaze_dfs = dataset_configuration['raw_gaze_dfs']
    for result_gaze_df, expected_gaze_df in zip(dataset.gaze, expected_gaze_dfs):
        assert_frame_equal(result_gaze_df.frame, expected_gaze_df.frame)


def test_load_gaze_has_position_columns(dataset_configuration):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True)

    for result_gaze_df in dataset.gaze:
        assert result_gaze_df.position_columns


def test_load_correct_preprocessed_gaze_dfs(dataset_configuration):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True)

    expected_gaze_dfs = dataset_configuration['preprocessed_gaze_dfs']
    for result_gaze_df, expected_gaze_df in zip(dataset.gaze, expected_gaze_dfs):
        assert_frame_equal(result_gaze_df.frame, expected_gaze_df.frame)


def test_load_correct_event_dfs(dataset_configuration):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load(events=True)

    expected_event_dfs = dataset_configuration['event_dfs']
    for result_event_df, expected_event_df in zip(dataset.events, expected_event_dfs):
        assert_frame_equal(result_event_df.frame, expected_event_df)


@pytest.mark.parametrize(
    ('subset', 'fileinfo_idx'),
    [
        pytest.param(
            {'subject_id': 1},
            [0],
            id='subset_int',
        ),
        pytest.param(
            {'subject_id': [1, 11, 12]},
            [0, 2, 3],
            id='subset_list',
        ),
        pytest.param(
            {'subject_id': range(3)},
            [0, 11],
            id='subset_range',
        ),
    ],
)
def test_load_subset(subset, fileinfo_idx, dataset_configuration):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load(subset=subset)

    expected_fileinfo = dataset_configuration['fileinfo']
    expected_fileinfo = expected_fileinfo[fileinfo_idx]

    assert_frame_equal(dataset.fileinfo, expected_fileinfo)


@pytest.mark.parametrize(
    ('init_kwargs', 'load_kwargs', 'exception'),
    [
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
    dataset = pm.Dataset(**init_kwargs)

    with pytest.raises(exception):
        dataset.load(**load_kwargs)


@pytest.mark.parametrize(
    ('init_kwargs', 'save_kwargs', 'exception'),
    [
        pytest.param(
            {},
            {'extension': 'invalid'},
            ValueError,
            id='wrong_extension_save_gaze',
        ),
    ],
)
def test_save_gaze_exceptions(init_kwargs, save_kwargs, exception, dataset_configuration):
    init_kwargs = {**dataset_configuration['init_kwargs'], **init_kwargs}
    dataset = pm.Dataset(**init_kwargs)

    with pytest.raises(exception):
        dataset.load()
        dataset.pix2deg()
        dataset.pos2vel()
        dataset.save_preprocessed(**save_kwargs)


@pytest.mark.parametrize(
    ('init_kwargs', 'load_kwargs', 'exception'),
    [
        pytest.param(
            {},
            {'extension': 'invalid'},
            ValueError,
            id='wrong_extension_load_events',
        ),
    ],
)
def test_load_events_exceptions(init_kwargs, load_kwargs, exception, dataset_configuration):
    init_kwargs = {**dataset_configuration['init_kwargs'], **init_kwargs}
    dataset = pm.Dataset(**init_kwargs)

    with pytest.raises(exception):
        dataset.load()
        dataset.pix2deg()
        dataset.pos2vel()
        dataset.detect_events(
            method=pm.events.ivt,
            velocity_threshold=45,
            minimum_duration=55,
        )
        dataset.save_events()
        dataset.load_event_files(**load_kwargs)


@pytest.mark.parametrize(
    ('init_kwargs', 'save_kwargs', 'exception'),
    [
        pytest.param(
            {},
            {'extension': 'invalid'},
            ValueError,
            id='wrong_extension_events',
        ),
    ],
)
def test_save_events_exceptions(init_kwargs, save_kwargs, exception, dataset_configuration):
    init_kwargs = {**dataset_configuration['init_kwargs'], **init_kwargs}
    dataset = pm.Dataset(**init_kwargs)

    with pytest.raises(exception):
        dataset.load()
        dataset.pix2deg()
        dataset.pos2vel()
        dataset.detect_events(
            method=pm.events.ivt,
            velocity_threshold=45,
            minimum_duration=55,
        )
        dataset.save_events(**save_kwargs)


def test_load_no_files_raises_exception(dataset_configuration):
    init_kwargs = {**dataset_configuration['init_kwargs']}
    dataset = pm.Dataset(**init_kwargs)

    shutil.rmtree(dataset.paths.raw, ignore_errors=True)
    dataset.paths.raw.mkdir()

    with pytest.raises(RuntimeError):
        dataset.load()


@pytest.mark.parametrize(
    'exception',
    [
        pytest.param(ValueError, id='matlab_dataset'),
    ],
)
@pytest.mark.parametrize('dataset_configuration', ['ToyMat'], indirect=['dataset_configuration'])
def test_load_mat_file_exception(exception, dataset_configuration):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])

    with pytest.raises(exception):
        dataset.load()


def test_pix2deg(dataset_configuration):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()

    original_schema = dataset.gaze[0].schema

    dataset.pix2deg()

    dva_schema = {}
    for column_name in original_schema.keys():
        if column_name.endswith('_pix'):
            dva_column_name = column_name.replace('_pix', '_pos')
            dva_schema[dva_column_name] = original_schema[column_name]
    expected_schema = {**original_schema, **dva_schema}

    for result_gaze_df in dataset.gaze:
        print(result_gaze_df.schema)
        print(expected_schema)
        assert result_gaze_df.schema == expected_schema


def test_pos2acc(dataset_configuration):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()
    dataset.pix2deg()

    original_schema = dataset.gaze[0].schema

    dataset.pos2acc()

    acc_schema = {}
    for column_name in original_schema.keys():
        if column_name.endswith('_pix'):
            acc_column_name = column_name.replace('_pix', '_acc')
            acc_schema[acc_column_name] = original_schema[column_name]
    expected_schema = {**original_schema, **acc_schema}

    for result_gaze_df in dataset.gaze:
        assert result_gaze_df.schema == expected_schema


def test_pos2vel(dataset_configuration):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()
    dataset.pix2deg()

    original_schema = dataset.gaze[0].schema

    dataset.pos2vel()

    vel_schema = {}
    for column_name in original_schema.keys():
        if column_name.endswith('_pix'):
            vel_column_name = column_name.replace('_pix', '_vel')
            vel_schema[vel_column_name] = original_schema[column_name]
    expected_schema = {**original_schema, **vel_schema}

    for result_gaze_df in dataset.gaze:
        assert result_gaze_df.schema == expected_schema


@pytest.mark.parametrize(
    'detect_event_kwargs',
    [
        pytest.param(
            {
                'method': pm.events.microsaccades,
                'threshold': 1,
                'eye': 'auto',
            },
            id='microsaccades_class',
        ),
        pytest.param(
            {
                'method': 'microsaccades',
                'threshold': 1,
                'eye': 'auto',
            },
            id='microsaccades_string',
        ),
        pytest.param(
            {
                'method': 'fill',
                'eye': 'auto',
            },
            id='fill_string',
        ),
    ],
)
def test_detect_events_auto_eye(detect_event_kwargs, dataset_configuration):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()
    dataset.pix2deg()
    dataset.pos2vel()
    dataset.detect_events(**detect_event_kwargs)

    expected_schema = {
        'subject_id': pl.Int64, **pm.events.EventDataFrame._minimal_schema, 'duration': pl.Int64,
    }
    for result_event_df in dataset.events:
        assert result_event_df.schema == expected_schema


@pytest.mark.parametrize(
    'detect_event_kwargs',
    [
        pytest.param(
            {
                'method': pm.events.microsaccades,
                'threshold': 1,
                'eye': 'left',
            },
            id='left',
        ),
        pytest.param(
            {
                'method': pm.events.microsaccades,
                'threshold': 1,
                'eye': 'right',
            },
            id='right',
        ),
        pytest.param(
            {
                'method': pm.events.microsaccades,
                'threshold': 1,
                'eye': 'eye',
            },
            id='eye',
        ),
    ],
)
def test_detect_events_explicit_eye(detect_event_kwargs, dataset_configuration):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()
    dataset.pix2deg()
    dataset.pos2vel()

    dataset_eyes = dataset_configuration['eyes']

    exception = None
    if dataset_eyes != detect_event_kwargs['eye']:
        if dataset_eyes != 'both' or detect_event_kwargs['eye'] == 'eye':
            exception = AttributeError

    if exception is None:
        dataset.detect_events(**detect_event_kwargs)

        expected_schema = {
            'subject_id': pl.Int64,
            **pm.events.EventDataFrame._minimal_schema,
            'duration': pl.Int64,
        }

        for result_event_df in dataset.events:
            assert result_event_df.schema == expected_schema

    else:
        with pytest.raises(exception):
            dataset.detect_events(**detect_event_kwargs)


@pytest.mark.parametrize(
    ('detect_event_kwargs_1', 'detect_event_kwargs_2', 'expected_schema'),
    [
        pytest.param(
            {
                'method': pm.events.microsaccades,
                'threshold': 1,
                'eye': 'auto',
            },
            {
                'method': pm.events.microsaccades,
                'threshold': 1,
                'eye': 'auto',
            },
            {
                'subject_id': pl.Int64,
                **pm.events.EventDataFrame._minimal_schema,
                'duration': pl.Int64,
            },
            id='two-saccade-runs',
        ),
        pytest.param(
            {
                'method': pm.events.microsaccades,
                'threshold': 1,
                'eye': 'auto',
            },
            {
                'method': pm.events.ivt,
                'velocity_threshold': 1,
                'minimum_duration': 1,
            },
            {
                'subject_id': pl.Int64,
                **pm.events.EventDataFrame._minimal_schema,
                'duration': pl.Int64,
            },
            id='one-saccade-one-fixation-run',
        ),
    ],
)
def test_detect_events_multiple_calls(
        detect_event_kwargs_1, detect_event_kwargs_2,
        expected_schema, dataset_configuration,
):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()
    dataset.pix2deg()
    dataset.pos2vel()
    dataset.detect_events(**detect_event_kwargs_1)
    dataset.detect_events(**detect_event_kwargs_2)

    for result_event_df in dataset.events:
        assert result_event_df.schema == expected_schema


@pytest.mark.parametrize(
    'detect_kwargs',
    [
        pytest.param(
            {
                'method': 'microsaccades',
                'threshold': 1,
                'eye': 'left',
                'clear': False,
                'verbose': True,
            },
            id='left',
        ),
    ],
)
def test_detect_events_alias(dataset_configuration, detect_kwargs, monkeypatch):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()
    dataset.pix2deg()
    dataset.pos2vel()

    mock = Mock()
    monkeypatch.setattr(dataset, 'detect_events', mock)

    dataset.detect(**detect_kwargs)
    mock.assert_called_with(**detect_kwargs)


def test_detect_events_attribute_error(dataset_configuration):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()

    try:
        dataset.gaze[0] = dataset.gaze[0].drop('x_left_pos')
    except BaseException:
        pass

    try:
        dataset.gaze[0] = dataset.gaze[0].drop('x_right_pos')
    except BaseException:
        pass

    try:
        dataset.gaze[0] = dataset.gaze[0].drop('x_eye_pos')
    except BaseException:
        pass

    detect_event_kwargs = {
        'method': pm.events.microsaccades,
        'threshold': 1,
        'eye': 'auto',
    }

    with pytest.raises(AttributeError):
        dataset.detect_events(**detect_event_kwargs)


@pytest.mark.parametrize(
    ('events_init', 'events_expected'),
    [
        pytest.param(
            [],
            [],
            id='empty_list_stays_empty_list',
        ),
        pytest.param(
            [pm.events.EventDataFrame()],
            [pm.events.EventDataFrame()],
            id='empty_df_stays_empty_df',
        ),
        pytest.param(
            [pm.events.EventDataFrame(name='event', onsets=[0], offsets=[99])],
            [pm.events.EventDataFrame()],
            id='single_instance_filled_df_gets_cleared_to_empty_df',
        ),
        pytest.param(
            [
                pm.events.EventDataFrame(name='event', onsets=[0], offsets=[99]),
                pm.events.EventDataFrame(name='event', onsets=[0], offsets=[99]),
            ],
            [pm.events.EventDataFrame(), pm.events.EventDataFrame()],
            id='two_instance_filled_df_gets_cleared_to_two_empty_dfs',
        ),
    ],
)
def test_clear_events(events_init, events_expected, tmp_path):
    dataset = pm.Dataset('ToyDataset', path=tmp_path)
    dataset.events = events_init
    dataset.clear_events()

    if isinstance(events_init, list) and not events_init:
        assert dataset.events == events_expected

    else:
        for events_df_result, events_df_expected in zip(dataset.events, events_expected):
            assert_frame_equal(events_df_result.frame, events_df_expected.frame)


@pytest.mark.parametrize(
    ('detect_event_kwargs', 'events_dirname', 'expected_save_dirpath', 'save_kwargs'),
    [
        pytest.param(
            {'method': pm.events.microsaccades, 'threshold': 1, 'eye': 'auto'},
            None,
            'events',
            {},
            id='none_dirname',
        ),
        pytest.param(
            {'method': pm.events.microsaccades, 'threshold': 1, 'eye': 'auto'},
            'events_test',
            'events_test',
            {},
            id='explicit_dirname',
        ),
        pytest.param(
            {'method': pm.events.microsaccades, 'threshold': 1, 'eye': 'auto'},
            None,
            'events',
            {'extension': 'csv'},
            id='save_events_extension_csv',
        ),
    ],
)
def test_save_events(
        detect_event_kwargs,
        events_dirname,
        expected_save_dirpath,
        save_kwargs,
        dataset_configuration,
):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()
    dataset.pix2deg()
    dataset.pos2vel()
    dataset.detect_events(**detect_event_kwargs)

    if events_dirname is None:
        events_dirname = 'events'
    shutil.rmtree(dataset.path / Path(events_dirname), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(expected_save_dirpath), ignore_errors=True)
    dataset.save_events(events_dirname, **save_kwargs)

    assert (dataset.path / expected_save_dirpath).is_dir(), (
        f'data was not written to {dataset.path / Path(expected_save_dirpath)}'
    )


@pytest.mark.parametrize(
    ('detect_event_kwargs', 'events_dirname', 'expected_save_dirpath', 'load_save_kwargs'),
    [
        pytest.param(
            {'method': pm.events.microsaccades, 'threshold': 1, 'eye': 'auto'},
            None,
            'events',
            {},
            id='none_dirname',
        ),
        pytest.param(
            {'method': pm.events.microsaccades, 'threshold': 1, 'eye': 'auto'},
            'events_test',
            'events_test',
            {},
            id='explicit_dirname',
        ),
        pytest.param(
            {'method': pm.events.microsaccades, 'threshold': 1, 'eye': 'auto'},
            None,
            'events',
            {'extension': 'csv'},
            id='load_events_extension_csv',
        ),
    ],
)
def test_load_previously_saved_events_gaze(
        detect_event_kwargs,
        events_dirname,
        expected_save_dirpath,
        load_save_kwargs,
        dataset_configuration,
):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()
    dataset.pix2deg()
    dataset.pos2vel()
    dataset.detect_events(**detect_event_kwargs)

    # We must not overwrite the original variable as it's needed in the end.
    if events_dirname is None:
        events_dirname_ = 'events'
    else:
        events_dirname_ = events_dirname

    shutil.rmtree(dataset.path / Path(events_dirname_), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(expected_save_dirpath), ignore_errors=True)
    dataset.save_events(events_dirname, **load_save_kwargs)
    dataset.save_preprocessed(**load_save_kwargs)

    dataset.events = []

    dataset.load(events=True, preprocessed=True, events_dirname=events_dirname, **load_save_kwargs)
    assert dataset.events


@pytest.mark.parametrize(
    ('preprocessed_dirname', 'expected_save_dirpath'),
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
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
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
    ('expected_save_preprocessed_path', 'expected_save_events_path', 'save_kwargs'),
    [
        pytest.param(
            'preprocessed',
            'events',
            {},
            id='none_dirname',
        ),
        pytest.param(
            'preprocessed',
            'events',
            {'verbose': 2},
            id='verbose_equals_2',
        ),
        pytest.param(
            'preprocessed_test',
            'events',
            {'preprocessed_dirname': 'preprocessed_test'},
            id='explicit_prepocessed_dirname',
        ),
        pytest.param(
            'preprocessed',
            'events_test',
            {'events_dirname': 'events_test'},
            id='explicit_events_dirname',
        ),
        pytest.param(
            'preprocessed',
            'events',
            {'extension': 'csv'},
            id='extension_equals_csv',
        ),
    ],
)
def test_save_creates_correct_directory(
        expected_save_preprocessed_path,
        expected_save_events_path,
        save_kwargs,
        dataset_configuration,
):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()
    dataset.pix2deg()
    dataset.pos2vel()

    detect_events_kwargs = {'method': pm.events.microsaccades, 'threshold': 1, 'eye': 'auto'}
    dataset.detect_events(**detect_events_kwargs)

    preprocessed_dirname = save_kwargs.get('preprocessed_dirname', 'preprocessed')
    events_dirname = save_kwargs.get('events_dirname', 'events')

    shutil.rmtree(dataset.path / Path(preprocessed_dirname), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(expected_save_preprocessed_path), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(events_dirname), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(expected_save_events_path), ignore_errors=True)
    dataset.save(**save_kwargs)

    assert (dataset.path / Path(expected_save_preprocessed_path)).is_dir(), (
        f'data was not written to {dataset.path / Path(expected_save_preprocessed_path)}'
    )
    assert (dataset.path / Path(expected_save_events_path)).is_dir(), (
        f'data was not written to {dataset.path / Path(expected_save_events_path)}'
    )


@pytest.mark.parametrize(
    ('expected_save_preprocessed_path', 'expected_save_events_path', 'save_kwargs'),
    [
        pytest.param(
            'preprocessed',
            'events',
            {'extension': 'feather'},
            id='extension_equals_feather',
        ),
        pytest.param(
            'preprocessed',
            'events',
            {'extension': 'csv'},
            id='extension_equals_csv',
        ),
    ],
)
def test_save_files_have_correct_extension(
        expected_save_preprocessed_path,
        expected_save_events_path,
        save_kwargs,
        dataset_configuration,
):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()
    dataset.pix2deg()
    dataset.pos2vel()

    detect_events_kwargs = {'method': pm.events.microsaccades, 'threshold': 1, 'eye': 'auto'}
    dataset.detect_events(**detect_events_kwargs)

    preprocessed_dirname = save_kwargs.get('preprocessed_dirname', 'preprocessed')
    events_dirname = save_kwargs.get('events_dirname', 'events')
    extension = save_kwargs.get('extension', 'feather')

    shutil.rmtree(dataset.path / Path(preprocessed_dirname), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(expected_save_preprocessed_path), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(events_dirname), ignore_errors=True)
    shutil.rmtree(dataset.path / Path(expected_save_events_path), ignore_errors=True)
    dataset.save(**save_kwargs)

    preprocessed_dir = dataset.path / Path(expected_save_preprocessed_path)
    preprocessed_file_list = os.listdir(preprocessed_dir)
    extension_list = [a.endswith(extension) for a in preprocessed_file_list]
    extension_sum = sum(extension_list)
    assert extension_sum == len(preprocessed_file_list), (
        f'not all preprocessed files created have correct extension {extension}'
    )

    events_dir = dataset.path / Path(expected_save_events_path)
    events_file_list = os.listdir(events_dir)
    extension_list = [a.endswith(extension) for a in events_file_list]
    extension_sum = sum(extension_list)
    assert extension_sum == len(events_file_list), (
        f'not all events files created have correct extension {extension}'
    )


@pytest.mark.parametrize(
    ('init_path', 'expected_paths'),
    [
        pytest.param(
            '/data/set/path',
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/'),
                'raw': Path('/data/set/path/raw'),
                'preprocessed': Path('/data/set/path/preprocessed'),
                'events': Path('/data/set/path/events'),
            },
        ),
        pytest.param(
            pm.DatasetPaths(root='/data/set/path', dataset='.'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/'),
                'raw': Path('/data/set/path/raw'),
                'preprocessed': Path('/data/set/path/preprocessed'),
                'events': Path('/data/set/path/events'),
            },
        ),
        pytest.param(
            pm.DatasetPaths(root='/data/set/path', dataset='dataset'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/dataset'),
                'raw': Path('/data/set/path/dataset/raw'),
                'preprocessed': Path('/data/set/path/dataset/preprocessed'),
                'events': Path('/data/set/path/dataset/events'),
            },
        ),
        pytest.param(
            pm.DatasetPaths(root='/data/set/path', dataset='dataset', events='custom_events'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/dataset'),
                'raw': Path('/data/set/path/dataset/raw'),
                'preprocessed': Path('/data/set/path/dataset/preprocessed'),
                'events': Path('/data/set/path/dataset/custom_events'),
            },
        ),
        pytest.param(
            pm.DatasetPaths(
                root='/data/set/path',
                dataset='dataset',
                preprocessed='custom_preprocessed',
            ),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/dataset'),
                'raw': Path('/data/set/path/dataset/raw'),
                'preprocessed': Path('/data/set/path/dataset/custom_preprocessed'),
                'events': Path('/data/set/path/dataset/events'),
            },
        ),
        pytest.param(
            pm.DatasetPaths(root='/data/set/path', dataset='dataset', raw='custom_raw'),
            {
                'root': Path('/data/set/path/'),
                'dataset': Path('/data/set/path/dataset'),
                'raw': Path('/data/set/path/dataset/custom_raw'),
                'preprocessed': Path('/data/set/path/dataset/preprocessed'),
                'events': Path('/data/set/path/dataset/events'),
            },
        ),
    ],
)
def test_paths(init_path, expected_paths):
    dataset = pm.Dataset('ToyDataset', path=init_path)

    assert dataset.paths.root == expected_paths['root']
    assert dataset.paths.dataset == expected_paths['dataset']
    assert dataset.path == expected_paths['dataset']
    assert dataset.paths.raw == expected_paths['raw']
    assert dataset.paths.preprocessed == expected_paths['preprocessed']
    assert dataset.paths.events == expected_paths['events']


@pytest.mark.parametrize(
    ('new_fileinfo', 'exception'),
    [
        pytest.param(None, AttributeError),
        pytest.param([], AttributeError),
    ],
)
def test_check_fileinfo(new_fileinfo, exception, tmp_path):
    dataset = pm.Dataset('ToyDataset', path=tmp_path)

    dataset.fileinfo = new_fileinfo

    with pytest.raises(exception):
        dataset._check_fileinfo()


@pytest.mark.parametrize(
    ('new_gaze', 'exception'),
    [
        pytest.param(None, AttributeError),
        pytest.param([], AttributeError),
    ],
)
def test_check_gaze_dataframe(new_gaze, exception, tmp_path):
    dataset = pm.Dataset('ToyDataset', path=tmp_path)

    dataset.gaze = new_gaze

    with pytest.raises(exception):
        dataset._check_gaze_dataframe()


@pytest.mark.parametrize('dataset_configuration', ['ToyBino'], indirect=['dataset_configuration'])
def test_check_experiment(dataset_configuration):
    dataset_configuration['init_kwargs']['definition'].experiment = None
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load()

    with pytest.raises(AttributeError):
        dataset.gaze[0]._check_experiment()


@pytest.mark.parametrize('dataset_configuration', ['ToyBino'], indirect=['dataset_configuration'])
def test_velocity_columns(dataset_configuration):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True)

    expected_velocity_columns = ['x_left_vel', 'y_left_vel', 'x_right_vel', 'y_right_vel']
    case = unittest.TestCase()

    for gaze_df in dataset.gaze:
        case.assertCountEqual(gaze_df.velocity_columns, expected_velocity_columns)


@pytest.mark.parametrize(
    ('property_kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {'event_properties': 'foo'},
            pm.exceptions.InvalidProperty,
            ('foo', 'invalid', 'valid', 'peak_velocity'),
            id='invalid_property',
        ),
    ],
)
def test_event_dataframe_add_property_raises_exceptions(
        dataset_configuration, property_kwargs, exception, msg_substrings,
):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True, events=True)

    with pytest.raises(exception) as excinfo:
        dataset.compute_event_properties(**property_kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    'property_kwargs',
    [
        pytest.param({'event_properties': 'peak_velocity'}, id='peak_velocity'),
    ],
)
def test_event_dataframe_add_property_has_expected_height(dataset_configuration, property_kwargs):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True, events=True)

    expected_heights = [len(event_df) for event_df in dataset.events]

    dataset.compute_event_properties(**property_kwargs)

    for events_df, expected_height in zip(dataset.events, expected_heights):
        assert events_df.frame.height == expected_height


@pytest.mark.parametrize(
    ('property_kwargs', 'expected_schema'),
    [
        pytest.param(
            {'event_properties': 'peak_velocity'},
            {
                'subject_id': pl.Int64,
                **pm.events.EventDataFrame._minimal_schema,
                'duration': pl.Int64,
                'peak_velocity': pl.Float64,
            },
            id='single_event_peak_velocity',
        ),
        pytest.param(
            {'event_properties': 'location'},
            {
                'subject_id': pl.Int64,
                **pm.events.EventDataFrame._minimal_schema,
                'duration': pl.Int64,
                'location': pl.List(pl.Float64),
            },
            id='single_event_position',
        ),
    ],
)
def test_event_dataframe_add_property_has_expected_schema(
        dataset_configuration, property_kwargs, expected_schema,
):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True, events=True)

    dataset.compute_event_properties(**property_kwargs)

    for events_df in dataset.events:
        if events_df.frame.schema != expected_schema:
            print(events_df.frame)
        assert events_df.frame.schema == expected_schema


@pytest.mark.parametrize(
    ('property_kwargs', 'expected_property_columns'),
    [
        pytest.param(
            {'event_properties': 'peak_velocity'},
            ['peak_velocity'],
            id='single_event_peak_velocity',
        ),
        pytest.param(
            {'event_properties': 'location'},
            ['location'],
            id='single_event_location',
        ),
        pytest.param(
            {'event_properties': 'location', 'name': 'fixation'},
            ['location'],
            id='single_event_location_name_fixation',
        ),
        pytest.param(
            {'event_properties': 'peak_velocity', 'name': 'sacc.*'},
            ['peak_velocity'],
            id='single_event_peak_velocity_regex_name_sacc',
        ),
    ],
)
def test_event_dataframe_add_property_effect_property_columns(
        dataset_configuration, property_kwargs, expected_property_columns,
):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True, events=True)

    dataset.compute_event_properties(**property_kwargs)

    for events_df in dataset.events:
        assert events_df.event_property_columns == expected_property_columns


@pytest.mark.parametrize(
    ('property_kwargs', 'exception', 'exception_msg'),
    [
        pytest.param(
            {'event_properties': 'peak_velocity', 'name': 'taccade'},
            RuntimeError, 'No events with name "taccade" found in data frame',
            id='name_missing',
        ),
    ],
)
def test_event_dataframe_add_property_raises_exception(
        dataset_configuration, property_kwargs, exception, exception_msg,
):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True, events=True)

    with pytest.raises(exception) as excinfo:
        dataset.compute_event_properties(**property_kwargs)

    msg, = excinfo.value.args
    assert msg == exception_msg


@pytest.mark.parametrize(
    'property_kwargs',
    [
        pytest.param(
            {'event_properties': 'peak_velocity'},
            id='single_event_peak_velocity',
        ),
        pytest.param(
            {'event_properties': 'location'},
            id='single_event_position',
        ),
        pytest.param(
            {'event_properties': 'location', 'name': 'fixation'},
            id='single_event_position_name_fixation',
        ),
    ],
)
def test_event_dataframe_add_property_does_not_change_length(
        dataset_configuration, property_kwargs,
):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True, events=True)

    lengths_pre = [len(events_df.frame) for events_df in dataset.events]
    dataset.compute_event_properties(**property_kwargs)
    lengths_post = [len(events_df.frame) for events_df in dataset.events]

    assert lengths_pre == lengths_post


@pytest.mark.parametrize(
    'property_kwargs',
    [
        pytest.param(
            {
                'event_properties': 'peak_velocity',
                'name': None,
                'verbose': True,
            }, id='peak_velocity',
        ),
    ],
)
def test_compute_event_properties_alias(dataset_configuration, property_kwargs, monkeypatch):
    dataset = pm.Dataset(**dataset_configuration['init_kwargs'])
    dataset.load(preprocessed=True, events=True)

    mock = Mock()
    monkeypatch.setattr(dataset, 'compute_event_properties', mock)

    dataset.compute_properties(**property_kwargs)
    mock.assert_called_with(**property_kwargs)
