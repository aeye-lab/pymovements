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
import numpy as np
import polars as pl
import pytest

import pymovements as pm


@pytest.fixture(
    name='gaze',
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
def fixture_gaze(request):
    init_param_dict = {
        'csv_monocular': {
            'definition': pm.dataset.DatasetDefinition(
                has_files={
                    'gaze': True,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
                time_column='time',
                time_unit='ms',
                pixel_columns=['x_left_pix', 'y_left_pix'],
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
                filename_format={'gaze': 'monocular_example.csv'},
                filename_format_schema_overrides={'gaze': {}},
                custom_read_kwargs={'gaze': {}},
            ),
            'filepath': 'tests/files/monocular_example.csv',
        },
        'csv_binocular': {
            'definition': pm.dataset.DatasetDefinition(
                has_files={
                    'gaze': True,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
                filename_format={'gaze': 'binocular_example.csv'},
                time_column='time',
                time_unit='ms',
                pixel_columns=['x_left_pix', 'y_left_pix', 'x_right_pix', 'y_right_pix'],
                position_columns=['x_left_pos', 'y_left_pos', 'x_right_pos', 'y_right_pos'],
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
                filename_format_schema_overrides={'gaze': {}},
                custom_read_kwargs={'gaze': {}},
            ),
            'filepath': 'tests/files/binocular_example.csv',
        },
        'ipc_monocular': {
            'definition': pm.dataset.DatasetDefinition(
                has_files={
                    'gaze': True,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
                filename_format={'gaze': 'monocular_example.feather'},
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
                filename_format_schema_overrides={'gaze': {}},
                custom_read_kwargs={'gaze': {}},
            ),
            'filepath': 'tests/files/monocular_example.feather',
        },
        'ipc_binocular': {
            'definition': pm.dataset.DatasetDefinition(
                has_files={
                    'gaze': True,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
                filename_format={'gaze': 'binocular_example.feather'},
                experiment=pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
                filename_format_schema_overrides={'gaze': {}},
                custom_read_kwargs={'gaze': {}},
            ),
            'filepath': 'tests/files/binocular_example.feather',
        },
        'emtec': {
            'definition': pm.datasets.EMTeC(
                has_files={
                    'gaze': True,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
                filename_format={'gaze': 'emtec_example.csv'},
                time_column=pm.datasets.EMTeC().time_column,
                time_unit=pm.datasets.EMTeC().time_unit,
                filename_format_schema_overrides={'gaze': {}},
                trial_columns=None,
            ),
            'filepath': 'tests/files/emtec_example.csv',
        },
        'didec': {
            'definition': pm.datasets.DIDEC(
                has_files={
                    'gaze': True,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
                filename_format={'gaze': 'didec_example.txt'},
                time_column=pm.datasets.DIDEC().time_column,
                time_unit=pm.datasets.DIDEC().time_unit,
                filename_format_schema_overrides={'gaze': {}},
                trial_columns=None,
            ),
            'filepath': 'tests/files/didec_example.txt',
        },
        'hbn': {
            'definition': pm.datasets.HBN(
                has_files={
                    'gaze': True,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
                filename_format={'gaze': 'hbn_example.csv'},
                time_column=pm.datasets.HBN().time_column,
                time_unit=pm.datasets.HBN().time_unit,
                filename_format_schema_overrides={'gaze': {}},
                trial_columns=None,
            ),
            'filepath': 'tests/files/hbn_example.csv',
        },
        'sbsat': {
            'definition': pm.datasets.SBSAT(
                has_files={
                    'gaze': True,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
                filename_format={'gaze': 'sbsat_example.csv'},
                time_column=pm.datasets.SBSAT().time_column,
                time_unit=pm.datasets.SBSAT().time_unit,
                filename_format_schema_overrides={'gaze': {}},
                trial_columns=None,
            ),
            'filepath': 'tests/files/sbsat_example.csv',
        },
        'gaze_on_faces': {
            'definition': pm.datasets.GazeOnFaces(
                has_files={
                    'gaze': True,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
                filename_format={'gaze': 'gaze_on_faces_example.csv'},
                time_column=pm.datasets.GazeOnFaces().time_column,
                time_unit=pm.datasets.GazeOnFaces().time_unit,
                filename_format_schema_overrides={'gaze': {}},
                trial_columns=None,
            ),
            'filepath': 'tests/files/gaze_on_faces_example.csv',
        },
        'gazebase': {
            'definition': pm.datasets.GazeBase(
                has_files={
                    'gaze': True,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
                filename_format={'gaze': 'gazebase_example.csv'},
                time_column=pm.datasets.GazeBase().time_column,
                time_unit=pm.datasets.GazeBase().time_unit,
                filename_format_schema_overrides={'gaze': {}},
                trial_columns=None,
            ),
            'filepath': 'tests/files/gazebase_example.csv',
        },
        'gazebase_vr': {
            'definition': pm.datasets.GazeBaseVR(
                has_files={
                    'gaze': True,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
                filename_format={'gaze': 'gazebase_vr_example.csv'},
                time_column=pm.datasets.GazeBaseVR().time_column,
                time_unit=pm.datasets.GazeBaseVR().time_unit,
                filename_format_schema_overrides={'gaze': {}},
                trial_columns=None,
            ),
            'filepath': 'tests/files/gazebase_vr_example.csv',
        },
        'gazegraph': {
            'definition': pm.datasets.GazeGraph(
                has_files={
                    'gaze': True,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
                filename_format={'gaze': 'gazegraph_example.csv'},
                time_column=pm.datasets.GazeGraph().time_column,
                time_unit=pm.datasets.GazeGraph().time_unit,
                filename_format_schema_overrides={'gaze': {}},
                trial_columns=None,
            ),
            'filepath': 'tests/files/gazegraph_example.csv',
        },
        'judo1000': {
            'definition': pm.datasets.JuDo1000(
                has_files={
                    'gaze': True,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
                filename_format={'gaze': 'judo1000_example.csv'},
                time_column=pm.datasets.JuDo1000().time_column,
                time_unit=pm.datasets.JuDo1000().time_unit,
                filename_format_schema_overrides={'gaze': {}},
                trial_columns=['trial_id'],
            ),
            'filepath': 'tests/files/judo1000_example.csv',
        },
        'potec': {
            'definition': pm.datasets.PoTeC(
                has_files={
                    'gaze': True,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
                filename_format={'gaze': 'potec_example.tsv'},
                time_column=pm.datasets.PoTeC().time_column,
                time_unit=pm.datasets.PoTeC().time_unit,
                filename_format_schema_overrides={'gaze': {}},
                trial_columns=None,
            ),
            'filepath': 'tests/files/potec_example.tsv',
        },
        'toy_monocular': {
            'definition': pm.DatasetDefinition(
                has_files={
                    'gaze': True,
                    'precomputed_events': False,
                    'precomputed_reading_measures': False,
                },
                trial_columns=['stimulus_id', 'trial_id'],
                pixel_columns=['x_pix', 'y_pix'],
            ),
            'samples': pl.from_dict(
                {
                    'time': np.arange(1000),
                    'x_pix': np.zeros(1000),
                    'y_pix': np.zeros(1000),
                    'stimulus_id': ['A'] * 500 + ['B'] * 500,
                    'trial_id': np.concatenate(
                        [
                            # stimulus A
                            np.zeros(100), np.ones(100), np.ones(100) * 2,
                            np.ones(100) * 3, np.ones(100) * 4,
                            # stimulus B
                            np.zeros(100), np.ones(100), np.ones(100) * 2,
                            np.ones(100) * 3, np.ones(100) * 4,
                        ],
                    ),
                },
                schema={
                    'time': pl.Int64,
                    'x_pix': pl.Float64,
                    'y_pix': pl.Float64,
                    'stimulus_id': pl.Utf8,
                    'trial_id': pl.Float64,
                },
            ),

        },
    }

    init_params = init_param_dict[request.param]

    definition = init_params['definition']

    if 'filepath' in init_params:
        filepath = init_params['filepath']

        if filepath.endswith('.csv') or filepath.endswith('.tsv') or filepath.endswith('.txt'):
            yield pm.gaze.from_csv(filepath, definition=definition)

        if filepath.endswith('.ipc') or filepath.endswith('.feather'):
            yield pm.gaze.from_ipc(filepath, definition=definition)

    if 'samples' in init_params:
        samples = init_params['samples']
        yield pm.GazeDataFrame(samples, definition=definition)
