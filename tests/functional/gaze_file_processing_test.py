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
"""Test basic preprocessing on various gaze files."""
import os.path

import pytest

from pymovements import datasets
from pymovements import Experiment
from pymovements import EyeTracker
from pymovements import gaze as gaze_module


@pytest.fixture(
    name='gaze_from_kwargs',
    params=[
        'csv_monocular',
        'csv_binocular',
        'ipc_monocular',
        'ipc_binocular',
        'eyelink_monocular',
        'eyelink_monocular_2khz',
        'eyelink_monocular_no_dummy',
        'didec',
        'emtec',
        'hbn',
        'sbsat',
        'gaze_on_faces',
        'gazebase',
        'gazebase_vr',
        'judo1000',
        'potec',
    ],
)
def fixture_gaze_init_kwargs(request):
    init_param_dict = {
        'csv_monocular': {
            'file': 'tests/files/monocular_example.csv',
            'time_column': 'time',
            'time_unit': 'ms',
            'pixel_columns': ['x_left_pix', 'y_left_pix'],
            'experiment': Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        },
        'csv_binocular': {
            'file': 'tests/files/binocular_example.csv',
            'time_column': 'time',
            'time_unit': 'ms',
            'pixel_columns': ['x_left_pix', 'y_left_pix', 'x_right_pix', 'y_right_pix'],
            'position_columns': ['x_left_pos', 'y_left_pos', 'x_right_pos', 'y_right_pos'],
            'experiment': Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        },
        'ipc_monocular': {
            'file': 'tests/files/monocular_example.feather',
            'experiment': Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        },
        'ipc_binocular': {
            'file': 'tests/files/binocular_example.feather',
            'experiment': Experiment(1024, 768, 38, 30, 60, 'center', 1000),
        },
        'eyelink_monocular': {
            'file': 'tests/files/eyelink_monocular_example.asc',
            'definition': datasets.ToyDatasetEyeLink(),
        },
        'eyelink_monocular_2khz': {
            'file': 'tests/files/eyelink_monocular_2khz_example.asc',
            'experiment': Experiment(
                1280, 1024, 38, 30.2, 68, 'upper left',
                eyetracker=EyeTracker(
                    sampling_rate=2000.0, left=True, right=False,
                    model='EyeLink Portable Duo', vendor='EyeLink',
                ),
            ),
        },
        'eyelink_monocular_no_dummy': {
            'file': 'tests/files/eyelink_monocular_no_dummy_example.asc',
            'experiment': Experiment(
                1920, 1080, 38, 30.2, 68, 'upper left',
                eyetracker=EyeTracker(
                    sampling_rate=500.0, left=True, right=False,
                    model='EyeLink 1000 Plus', vendor='EyeLink',
                ),
            ),
        },
        'didec': {
            'file': 'tests/files/didec_example.txt',
            'definition': datasets.DIDEC(),
        },
        'emtec': {
            'file': 'tests/files/emtec_example.csv',
            'definition': datasets.EMTeC(),
        },
        'hbn': {
            'file': 'tests/files/hbn_example.csv',
            'definition': datasets.HBN(),
        },
        'sbsat': {
            'file': 'tests/files/sbsat_example.csv',
            'definition': datasets.SBSAT(),
        },
        'gaze_on_faces': {
            'file': 'tests/files/gaze_on_faces_example.csv',
            'definition': datasets.GazeOnFaces(),
        },
        'gazebase': {
            'file': 'tests/files/gazebase_example.csv',
            'definition': datasets.GazeBase(),
        },
        'gazebase_vr': {
            'file': 'tests/files/gazebase_vr_example.csv',
            'definition': datasets.GazeBaseVR(),
        },
        'judo1000': {
            'file': 'tests/files/judo1000_example.csv',
            'definition': datasets.JuDo1000(),
        },
        'potec': {
            'file': 'tests/files/potec_example.tsv',
            'definition': datasets.PoTeC(),
        },

    }
    yield init_param_dict[request.param]


def test_gaze_file_processing(gaze_from_kwargs):
    # Load in gaze file.
    file_extension = os.path.splitext(gaze_from_kwargs['file'])[1]
    gaze = None
    if file_extension in {'.csv', '.tsv', '.txt'}:
        gaze = gaze_module.from_csv(**gaze_from_kwargs)
    elif file_extension in {'.feather', '.ipc'}:
        gaze = gaze_module.from_ipc(**gaze_from_kwargs)
    elif file_extension == '.asc':
        gaze = gaze_module.from_asc(**gaze_from_kwargs)

    assert gaze is not None
    assert gaze.frame.height > 0

    # Do some basic transformations.
    if 'pixel' in gaze.columns:
        gaze.pix2deg()
    gaze.pos2vel()
    gaze.pos2acc()
    gaze.resample(resampling_rate=2000)

    assert 'position' in gaze.columns
    assert 'velocity' in gaze.columns
    assert 'acceleration' in gaze.columns
