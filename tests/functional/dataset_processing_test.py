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
"""Test basic preprocessing on various datasets."""
import pytest

import pymovements as pm


@pytest.fixture(
    name='datasets',
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
    ],
)
def fixture_dataset_init_kwargs(request):
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
            trial_columns=None,
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


def test_dataset_save_load_preprocessed(datasets):
    dataset = datasets
    dataset.load()

    if 'pixel' in dataset.gaze[0].frame.columns:
        dataset.pix2deg()

    dataset.pos2vel()
    dataset.resample(resampling_rate=2000)
    dataset.save()
    dataset.load(preprocessed=True)
