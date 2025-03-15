# Copyright (c) 2023-2025 The pymovements Project Authors
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
"""Fixtures for datasets."""
import pytest

import pymovements as pm


@pytest.fixture(
    name='dataset',
    params=[
        'csv_monocular',
        'csv_binocular',
        'ipc_monocular',
        'ipc_binocular',
        'hbn',
        'sbsat',
        'gaze_on_faces',
        'gazebase',
        'gazebase_vr',
        'gazegraph',
        'judo1000',
        'potec',
        'toy_monocular',
        'toy_binocular',
        'toy_left_eye',
        'toy_right_eye',
        'toy_binocular_cyclopean',
        'toy_remote',
    ],
)
def fixture_dataset(request):
    yield pm.Dataset()

    '''
    init_param_dict = {
        'csv_monocular': pm.dataset.DatasetDefinition(
            time_column='time',
            time_unit='ms',
            pixel_columns=['x_left_pix', 'y_left_pix'],
            experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
            filename_format='monocular_example.csv',
        ),
        'csv_binocular': pm.dataset.DatasetDefinition(
            filename_format='binocular_example.csv',
            time_column='time',
            time_unit='ms',
            pixel_columns=['x_left_pix', 'y_left_pix', 'x_right_pix', 'y_right_pix'],
            position_columns=['x_left_pos', 'y_left_pos', 'x_right_pos', 'y_right_pos'],
            experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        ),
        'ipc_monocular': pm.dataset.DatasetDefinition(
            filename_format='monocular_example.feather',
            experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        ),
        'ipc_binocular': pm.dataset.DatasetDefinition(
            filename_format='binocular_example.feather',
            experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        ),
        'hbn': pm.datasets.HBN(
            filename_format='hbn_example.csv',
            time_column=pm.datasets.HBN().time_column,
            time_unit=pm.datasets.HBN().time_unit,
            filename_format_dtypes={},
            trial_columns=None,
        ),
        'sbsat': pm.datasets.SBSAT(
            filename_format='sbsat_example.csv',
            time_column=pm.datasets.SBSAT().time_column,
            time_unit=pm.datasets.SBSAT().time_unit,
            filename_format_dtypes={},

        ),
        'gaze_on_faces': pm.datasets.GazeOnFaces(
            filename_format='gaze_on_faces_example.csv',
            time_column=pm.datasets.GazeOnFaces().time_column,
            time_unit=pm.datasets.GazeOnFaces().time_unit,
            filename_format_dtypes={},
            trial_columns=None,
        ),
        'gazebase': pm.datasets.GazeBase(
            filename_format='gazebase_example.csv',
            time_column=pm.datasets.GazeBase().time_column,
            time_unit=pm.datasets.GazeBase().time_unit,
            filename_format_dtypes={},
            trial_columns=None,
        ),
        'gazebase_vr': pm.datasets.GazeBaseVR(
            filename_format='gazebase_vr_example.csv',
            time_column=pm.datasets.GazeBaseVR().time_column,
            time_unit=pm.datasets.GazeBaseVR().time_unit,
            filename_format_dtypes={},
            trial_columns=None,
        ),
        'gazegraph': pm.datasets.GazeGraph(
            filename_format='gazegraph_example.csv',
            time_column=pm.datasets.GazeGraph().time_column,
            time_unit=pm.datasets.GazeGraph().time_unit,
            filename_format_dtypes={},
            trial_columns=None,
        ),
        'judo1000': pm.datasets.JuDo1000(
            filename_format='judo1000_example.csv',
            time_column=pm.datasets.JuDo1000().time_column,
            time_unit=pm.datasets.JuDo1000().time_unit,
            filename_format_dtypes={},
            trial_columns=['trial_id'],
        ),
        'potec': pm.datasets.PoTeC(
            filename_format='potec_example.tsv',
            time_column=pm.datasets.PoTeC().time_column,
            time_unit=pm.datasets.PoTeC().time_unit,
            filename_format_dtypes={},
            trial_columns=None,
        ),

    }
    yield pm.dataset.Dataset(
        definition=init_param_dict[request.param],
        path=pm.dataset.DatasetPaths(
            root='tests',
            dataset='.',
            raw='files',
        ),
    )
    '''


'''



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
                gaze_df = gaze_df.frame.drop(key)

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


def create_precomputed_files_from_fileinfo(precomputed_dfs, fileinfo, rootpath):
    rootpath.mkdir(parents=True, exist_ok=True)

    for precomputed_df, fileinfo_row in zip(precomputed_dfs, fileinfo.to_dicts()):
        filepath = fileinfo_row['filepath']

        for key in fileinfo_row.keys():
            if key in precomputed_df.columns:
                precomputed_df = precomputed_df.drop(key)

        precomputed_df.write_csv(rootpath / filepath)


def create_precomputed_rm_files_from_fileinfo(precomputed_rm_df, fileinfo, rootpath):
    rootpath.mkdir(parents=True, exist_ok=True)

    for precomputed_rm_df, fileinfo_row in zip(precomputed_rm_df, fileinfo.to_dicts()):
        filepath = fileinfo_row['filepath']

        for key in fileinfo_row.keys():
            if key in precomputed_rm_df.columns:
                precomputed_rm_df = precomputed_rm_df.drop(key)

        precomputed_rm_df.write_csv(rootpath / filepath)


def mock_toy(
        rootpath,
        raw_fileformat,
        eyes,
        remote=False,
        has_files={
            'gaze': True,
            'precomputed_events': False,
            'precomputed_reading_measures': False,
        },
        extract={'gaze': True, 'precomputed_events': True},
        filename_format_schema_overrides={
            'gaze': {'subject_id': pl.Int64},
            'precomputed_events': {'subject_id': pl.Int64},
            'precomputed_reading_measures': {'subject_id': pl.Int64},
        },
):

    if filename_format_schema_overrides['precomputed_events']:
        subject_ids = list(range(1, 21))
        fileinfo = pl.DataFrame(
            data={'subject_id': subject_ids},
            schema={'subject_id': pl.Int64},
        )
    else:
        subject_ids = [str(idx) for idx in range(1, 21)]
        fileinfo = pl.DataFrame(
            data={'subject_id': subject_ids},
            schema={'subject_id': pl.Utf8},
        )

    fileinfo = fileinfo.with_columns([
        pl.format('{}.' + raw_fileformat, 'subject_id').alias('filepath'),
    ])

    fileinfo = fileinfo.sort(by='filepath')

    gaze_dfs = []
    for fileinfo_row in fileinfo.to_dicts():  # pylint: disable=not-an-iterable
        if eyes == 'both':
            gaze_df = pl.from_dict(
                {
                    'subject_id': fileinfo_row['subject_id'],
                    'time': np.arange(1000),
                    'x_left_pix': np.zeros(1000),
                    'y_left_pix': np.zeros(1000),
                    'x_right_pix': np.zeros(1000),
                    'y_right_pix': np.zeros(1000),
                    'trial_id_1': np.concatenate([np.zeros(500), np.ones(500)]),
                    'trial_id_2': ['a'] * 200 + ['b'] * 200 + ['c'] * 600,
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'x_left_pix': pl.Float64,
                    'y_left_pix': pl.Float64,
                    'x_right_pix': pl.Float64,
                    'y_right_pix': pl.Float64,
                    'trial_id_1': pl.Float64,
                    'trial_id_2': pl.Utf8,
                },
            )
            pixel_columns = ['x_left_pix', 'y_left_pix', 'x_right_pix', 'y_right_pix']

        elif eyes == 'both+avg':
            gaze_df = pl.from_dict(
                {
                    'subject_id': fileinfo_row['subject_id'],
                    'time': np.arange(1000),
                    'x_left_pix': np.zeros(1000),
                    'y_left_pix': np.zeros(1000),
                    'x_right_pix': np.zeros(1000),
                    'y_right_pix': np.zeros(1000),
                    'x_avg_pix': np.zeros(1000),
                    'y_avg_pix': np.zeros(1000),
                    'trial_id_1': np.concatenate([np.zeros(500), np.ones(500)]),
                    'trial_id_2': ['a'] * 200 + ['b'] * 200 + ['c'] * 600,
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'x_left_pix': pl.Float64,
                    'y_left_pix': pl.Float64,
                    'x_right_pix': pl.Float64,
                    'y_right_pix': pl.Float64,
                    'x_avg_pix': pl.Float64,
                    'y_avg_pix': pl.Float64,
                    'trial_id_1': pl.Float64,
                    'trial_id_2': pl.Utf8,
                },
            )
            pixel_columns = [
                'x_left_pix', 'y_left_pix', 'x_right_pix', 'y_right_pix', 'x_avg_pix', 'y_avg_pix',
            ]

        elif eyes == 'left':
            gaze_df = pl.from_dict(
                {
                    'subject_id': fileinfo_row['subject_id'],
                    'time': np.arange(1000),
                    'x_left_pix': np.zeros(1000),
                    'y_left_pix': np.zeros(1000),
                    'trial_id_1': np.concatenate([np.zeros(500), np.ones(500)]),
                    'trial_id_2': ['a'] * 200 + ['b'] * 200 + ['c'] * 600,
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'x_left_pix': pl.Float64,
                    'y_left_pix': pl.Float64,
                    'trial_id_1': pl.Float64,
                    'trial_id_2': pl.Utf8,
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
                    'trial_id_1': np.concatenate([np.zeros(500), np.ones(500)]),
                    'trial_id_2': ['a'] * 200 + ['b'] * 200 + ['c'] * 600,
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'x_right_pix': pl.Float64,
                    'y_right_pix': pl.Float64,
                    'trial_id_1': pl.Float64,
                    'trial_id_2': pl.Utf8,
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
                    'trial_id_1': np.concatenate([np.zeros(500), np.ones(500)]),
                    'trial_id_2': ['a'] * 200 + ['b'] * 200 + ['c'] * 600,
                },
                schema={
                    'subject_id': pl.Int64,
                    'time': pl.Int64,
                    'x_pix': pl.Float64,
                    'y_pix': pl.Float64,
                    'trial_id_1': pl.Float64,
                    'trial_id_2': pl.Utf8,
                },
            )
            pixel_columns = ['x_pix', 'y_pix']
        else:
            raise ValueError(f'invalid value for eyes: {eyes}')

        if remote:
            gaze_df = gaze_df.with_columns([
                pl.lit(680.).alias('distance'),
            ])

            distance_column = 'distance'
            distance_cm = None
        else:
            distance_column = None
            distance_cm = 68

        gaze_dfs.append(gaze_df)

    create_raw_gaze_files_from_fileinfo(gaze_dfs, fileinfo, rootpath / 'raw')

    # Create GazeDataFrames for passing as ground truth
    gaze_dfs = [
        pm.GazeDataFrame(gaze_df, pixel_columns=pixel_columns)
        for gaze_df in gaze_dfs
    ]

    preprocessed_gaze_dfs = []
    for fileinfo_row in fileinfo.to_dicts():  # pylint: disable=not-an-iterable
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

        # Create GazeDataFrames for passing as ground truth
        gaze_df = pm.GazeDataFrame(
            pl.from_dict(gaze_data, schema=gaze_schema),
            pixel_columns=pixel_columns,
            position_columns=position_columns,
            velocity_columns=velocity_columns,
            acceleration_columns=acceleration_columns,
        )

        preprocessed_gaze_dfs.append(gaze_df)

    create_preprocessed_gaze_files_from_fileinfo(
        preprocessed_gaze_dfs, fileinfo, rootpath / 'preprocessed',
    )

    event_dfs = []
    for fileinfo_row in fileinfo.to_dicts():  # pylint: disable=not-an-iterable
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
            distance_cm=distance_cm,
            origin='upper left',
            sampling_rate=1000,
        ),
        filename_format={
            'gaze': r'{subject_id:d}.' + raw_fileformat,
            'precomputed_events': r'{subject_id:d}.' + raw_fileformat,
            'precomputed_reading_measures': r'{subject_id:d}.' + raw_fileformat,
        },
        filename_format_schema_overrides=filename_format_schema_overrides,
        custom_read_kwargs={
            'gaze': {},
            'precomputed_events': {},
            'precomputed_reading_measures': {},
        },
        time_column='time',
        time_unit='ms',
        distance_column=distance_column,
        pixel_columns=pixel_columns,
        has_files=has_files,
        extract=extract,
    )

    precomputed_dfs = []
    for fileinfo_row in fileinfo.to_dicts():  # pylint: disable=not-an-iterable
        precomputed_event_df = pl.from_dict(
            {
                'subject_id': fileinfo_row['subject_id'],
                'CURRENT_FIXATION_DURATION': np.arange(1000),
                'CURRENT_FIX_X': np.zeros(1000),
                'CURRENT_FIX_Y': np.zeros(1000),
                'trial_id_1': np.concatenate([np.zeros(500), np.ones(500)]),
                'trial_id_2': ['a'] * 200 + ['b'] * 200 + ['c'] * 600,
            },
            schema={
                'subject_id': pl.Int64,
                'CURRENT_FIXATION_DURATION': pl.Float64,
                'CURRENT_FIX_X': pl.Float64,
                'CURRENT_FIX_Y': pl.Float64,
                'trial_id_1': pl.Float64,
                'trial_id_2': pl.Utf8,
            },
        )
        precomputed_dfs.append(precomputed_event_df)

    create_precomputed_files_from_fileinfo(
        precomputed_dfs,
        fileinfo,
        rootpath / 'precomputed_events',
    )

    precomputed_rm_dfs = []
    for fileinfo_row in fileinfo.to_dicts():  # pylint: disable=not-an-iterable
        precomputed_rm_df = pl.from_dict(
            {
                'subject_id': fileinfo_row['subject_id'],
                'number_fix': np.arange(1000),
                'mean_fix_dur': np.zeros(1000),
            },
            schema={
                'subject_id': pl.Int64,
                'number_fix': pl.Float64,
                'mean_fix_dur': pl.Float64,
            },
        )
        precomputed_rm_dfs.append(precomputed_rm_df)

    create_precomputed_rm_files_from_fileinfo(
        precomputed_rm_dfs,
        fileinfo,
        rootpath / 'precomputed_reading_measures',
    )

    return {
        'init_kwargs': {
            'definition': dataset_definition,
            'path': pm.DatasetPaths(root=rootpath, dataset='.'),
        },
        'fileinfo': {
            'gaze': fileinfo,
            'precomputed_events': fileinfo,
            'precomputed_reading_measures': fileinfo,
        },
        'raw_gaze_dfs': gaze_dfs,
        'preprocessed_gaze_dfs': preprocessed_gaze_dfs,
        'event_dfs': event_dfs,
        'precomputed_rm_dfs': precomputed_rm_dfs,
        'eyes': eyes,
        'trial_columns': ['subject_id'],
    }


@pytest.fixture(
    name='gaze_dataset_configuration',
    params=[
        'ToyMono',
        'ToyBino',
        'ToyLeft',
        'ToyRight',
        'ToyBino+Avg',
        'ToyRemote',
    ],
)
def gaze_fixture_dataset(request, tmp_path):
    rootpath = tmp_path

    dataset_type = request.param
    if dataset_type == 'ToyBino':
        dataset_dict = mock_toy(rootpath, raw_fileformat='csv', eyes='both')
    elif dataset_type == 'ToyBino+Avg':
        dataset_dict = mock_toy(rootpath, raw_fileformat='csv', eyes='both+avg')
    elif dataset_type == 'ToyMono':
        dataset_dict = mock_toy(rootpath, raw_fileformat='csv', eyes='none')
    elif dataset_type == 'ToyLeft':
        dataset_dict = mock_toy(rootpath, raw_fileformat='csv', eyes='left')
    elif dataset_type == 'ToyRight':
        dataset_dict = mock_toy(rootpath, raw_fileformat='csv', eyes='right')
    elif dataset_type == 'ToyMat':
        dataset_dict = mock_toy(rootpath, raw_fileformat='mat', eyes='both')
    elif dataset_type == 'ToyRemote':
        dataset_dict = mock_toy(rootpath, raw_fileformat='csv', eyes='both', remote=True)
    else:
        raise ValueError(f'{request.param} not supported as dataset mock')

    yield dataset_dict


'''
