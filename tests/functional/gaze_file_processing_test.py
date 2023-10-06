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
"""Test basic preprocessing on various gaze files."""
import os.path

import pytest

import pymovements as pm


@pytest.fixture(
    name='gaze_from_kwargs',
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
        'judo1000',
    ],
)
def fixture_gaze_init_kwargs(request):
    init_param_dict = {
        'csv_monocular': {
            'file': 'tests/files/monocular_example.csv',
            'time_column': 'time', 'pixel_columns': ['x_left_pix', 'y_left_pix'],
            'experiment': pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        },
        'csv_binocular': {
            'file': 'tests/files/binocular_example.csv',
            'time_column': 'time',
            'pixel_columns': ['x_left_pix', 'y_left_pix', 'x_right_pix', 'y_right_pix'],
            'position_columns': ['x_left_pos', 'y_left_pos', 'x_right_pos', 'y_right_pos'],
            'experiment': pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        },
        'ipc_monocular': {
            'file': 'tests/files/monocular_example.feather',
            'experiment': pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        },
        'ipc_binocular': {
            'file': 'tests/files/binocular_example.feather',
            'experiment': pm.Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        },
        'hbn': {
            'file': 'tests/files/hbn_example.csv',
            'time_column': pm.datasets.HBN().time_column,
            'pixel_columns': pm.datasets.HBN().pixel_columns,
            'experiment': pm.datasets.HBN().experiment,
        },
        'sbsat': {
            'file': 'tests/files/sbsat_example.csv',
            'time_column': pm.datasets.SBSAT().time_column,
            'pixel_columns': pm.datasets.SBSAT().pixel_columns,
            'experiment': pm.datasets.SBSAT().experiment,
            **pm.datasets.SBSAT().custom_read_kwargs,
        },
        'gaze_on_faces': {
            'file': 'tests/files/gaze_on_faces_example.csv',
            'time_column': pm.datasets.GazeOnFaces().time_column,
            'pixel_columns': pm.datasets.GazeOnFaces().pixel_columns,
            'experiment': pm.datasets.GazeOnFaces().experiment,
            **pm.datasets.GazeOnFaces().custom_read_kwargs,
        },
        'gazebase': {
            'file': 'tests/files/gazebase_example.csv',
            'time_column': pm.datasets.GazeBase().time_column,
            'position_columns': pm.datasets.GazeBase().position_columns,
            'experiment': pm.datasets.GazeBase().experiment,
        },
        'gazebase_vr': {
            'file': 'tests/files/gazebase_vr_example.csv',
            'time_column': pm.datasets.GazeBaseVR().time_column,
            'position_columns': pm.datasets.GazeBaseVR().position_columns,
            'experiment': pm.datasets.GazeBaseVR().experiment,
        },
        'judo1000': {
            'file': 'tests/files/judo1000_example.csv',
            'time_column': pm.datasets.JuDo1000().time_column,
            'pixel_columns': pm.datasets.JuDo1000().pixel_columns,
            'experiment': pm.datasets.JuDo1000().experiment,
            **pm.datasets.JuDo1000().custom_read_kwargs,
        },

    }
    yield init_param_dict[request.param]


def test_gaze_file_processing(gaze_from_kwargs):
    # Load in gaze file.
    file_extension = os.path.splitext(gaze_from_kwargs['file'])[1]
    gaze = None
    if file_extension in {'.txt', '.csv'}:
        gaze = pm.gaze.from_csv(**gaze_from_kwargs)
    elif file_extension in {'.feather', '.ipc'}:
        gaze = pm.gaze.from_ipc(**gaze_from_kwargs)

    assert gaze is not None

    # Do some basic transformations.
    if 'pixel' in gaze.columns:
        gaze.pix2deg()
    gaze.pos2vel()
    gaze.pos2acc()

    assert 'position' in gaze.columns
    assert 'velocity' in gaze.columns
    assert 'acceleration' in gaze.columns
